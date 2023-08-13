import json
import csv
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np

from shared.data_structures import MyData
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder, getlabel_embedding
from entity.models import EntityModel_label_RICON  # rz+

from entity.utils_adversarial import FGM, PGD

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def compare_predictions(pred_results, output_file):
    incorrect_predictions = []
    for pred_result in pred_results:
        predicted_ner = pred_result['predicted_ner']  # 预测结果
        label = pred_result['label']  # 真实结果
        for pred_ner in predicted_ner:
            if pred_ner not in label:
                incorrect_predictions.append(pred_result)
                break

    with open(output_file, 'w', encoding='utf-8') as f:
        for pred_result in incorrect_predictions:
            f.write(json.dumps(pred_result, cls=NpEncoder, ensure_ascii=False) + '\n')
    logger.info('incorrect predictions to %s..' % (output_file))


def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...' % (args.output_dir))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)
    # # 获取模型参数字典
    # state_dict = model.bert_model.state_dict()
    #
    # # 保存模型参数字典
    # torch.save(state_dict, '{}/{}.pth'.format(args.output_dir, args.task))


def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict, _ = model.run_batch(batches[i], training=False, label_batch=label_batch, args=args)
        pred_ner = output_dict['pred_ner']  # 32句话的
        idx = i * len(batches[i])
        for k, (sample, preds) in enumerate(zip(batches[i], pred_ner)):  # 拿出1句
            ner_result[idx + k] = []
            for span, pred in zip(sample['spans'], preds):  # 第1句话中第1个候选span与该位置的预测值
                if pred == 0:  # 如果预测为0（即"其他"类）
                    continue
                ner_result[idx + k].append((ner_id2label[pred], span[0], span[1]))  # 实体预测值

            tot_pred_ett += len(ner_result[idx + k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for k, doc in enumerate(js):
        doc["predicted_ner"] = []
        for i in range(len(ner_result[k])):
            doc["predicted_ner"].append(ner_result[k][i])

        js[k] = doc

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder, ensure_ascii=False) for doc in js))
    incorrect_predictions_file = os.path.join(args.output_dir,
                                              args.test_incorrect_filename)  # 'incorrect_predictions.json'
    compare_predictions(js, incorrect_predictions_file)


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('\n Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0

    for i in tqdm(range(len(batches))):
        output_dict, _ = model.run_batch(batches[i], training=False, label_batch=label_batch)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1

    acc = l_cor / l_tot  # 这个准确率是包含O类的准确率，因此很高
    logger.info('Accuracy: %5f' % acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d' % (cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0  # 精确率：正确预测的实体数占预测的实体数的比例
    r = cor / tot_gold if cor > 0 else 0.0  # 召回率：正确预测的实体数占真实的实体数的比例
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f' % (p, r, f1))
    logger.info('Used time: %f' % (time.time() - c_time))
    return f1


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True,
                        choices=['ace04', 'ace05', 'scierc', 'msra', 'onto4', 'FOOD', 'weibo', 'RISK', 'resume', 'RMRB',
                                 'cluener', 'ALL', 'newall', 'news', 'HRD', 'RMRB2014', 'rd'])

    parser.add_argument('--data_dir', type=str, default=None, required=True,  # data/ace05/
                        help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output',
                        help="output directory of the entity model")

    #####
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'], help="fgm adversarial training")
    parser.add_argument("--loss_type", default="ce", choices=["ce", "focal", "lsce"], type=str, )

    parser.add_argument('--max_span_length', type=int, default=16,
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--label_length', type=int, default=10,
                        help="LST length")
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=1e-4,
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=100,
                        help="how often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=int, default=1,
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument('--do_train', action='store_true',
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_true',
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true',
                        help="whether to evaluate on test set")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json",
                        help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json",
                        help="the prediction filename for the test set")
    parser.add_argument('--test_incorrect_filename', type=str, default="incorrect_predictions.json",
                        help="the incorrect predictions filename for the test set")

    parser.add_argument('--use_albert', action='store_true',
                        help="whether to use ALBERT model")
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None,
                        help="the base model directory")

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--context_window', type=int, required=True, default=None,
                        help="the context window size W for the entity model")
    # rz+
    parser.add_argument('--ner_label_type', type=str, required=True, default=None,
                        help="the context window size W for the entity model")  # ner_label_type.json
    # rz+

    # RICON+
    parser.add_argument(
        "--ricon_hidden_size",
        default=200,
        type=int, )  # ricon模式下的输出隐藏层数目，为什么是200？？？
    parser.add_argument(
        "--ngram",
        type=int,
        # default=-1,  # -1 or 0 means dont slide
        default=16,  # -1 or 0 means dont slide
        help="ngram slide.", )  # 设置为16
    parser.add_argument(
        "--orth_loss_eof",
        type=float,
        default=0.0,
        help="orth_loss_eof.", )
    parser.add_argument(
        "--aware_loss_eof",
        type=float,
        default=1.0,
        help="aware_loss_eof.", )
    parser.add_argument(
        "--agnostic_loss_eof",
        type=float,
        default=0.0,
        help="agnostic_loss_eof.", )
    parser.add_argument(
        "--combination",
        type=str,
        default="x,y",
        # default="x,y,x*y",
        help="span combination.", )
    parser.add_argument("--multilabel_loss", action="store_true")
    # RICON+

    args = parser.parse_args()
    args.train_data = os.path.join(args.data_dir, 'train.json')
    # args.train_data = os.path.join(args.data_dir, 'train.tsv')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    # args.dev_data = os.path.join(args.data_dir, 'dev.tsv')
    # args.test_data = os.path.join(args.data_dir, 'test.json')
    # args.test_data = os.path.join(args.data_dir, 'test.tsv')

    if 'albert' in args.model:  # 跳过
        logger.info('Use Albert: %s' % args.model)
        args.use_albert = True

    setseed(args.seed)

    if not os.path.exists(args.output_dir):  # 'out/entity/'
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))  # 输出文件夹生成训练日志
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)

    # rz+
    from transformers import BertTokenizer, BertPreTrainedModel, BertModel

    # from transformers import AlbertTokenizer, AlbertPreTrainedModel, AlbertModel
    tokenizers = BertTokenizer.from_pretrained(args.bert_model_dir)
    label_batch = getlabel_embedding(args, tokenizers)
    # rz+

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])

    num_ner_labels = len(task_ner_labels[args.task]) + 1  # 取task_ner_labels列表中的ace05数据集标签数+1=8
    # model = EntityModel(args, num_ner_labels=num_ner_labels)  # EntityModel
    # rz+
    # model = EntityModel_label(args, num_ner_labels=num_ner_labels)
    model = EntityModel_label_RICON(args)  # 去掉参数num_ner_labels，这样模型可以给不同的数据集迁移，以实现few-shot


    # rz+

    def tsv2json(path):
        import ast
        if path.endswith('.tsv'):
            output_file = os.path.splitext(os.path.basename(path))[0] + '.json'
            tsv_file = open(path, "r", encoding='utf-8')
            json_file = open(os.path.join(os.path.dirname(path), output_file), "w", encoding='utf-8')

            field_names = ("text", "label")
            reader = csv.DictReader(tsv_file, field_names, delimiter="\t")

            # Skip the header row
            next(reader)

            json_list = []
            for row in reader:
                # 去掉text字段中的空格
                row['text'] = row['text'].replace(' ', '')
                # 将label字符串转为列表格式
                row['label'] = ast.literal_eval(row['label'])
                json_list.append(row)

            json.dump(json_list, json_file, ensure_ascii=False,
                      indent=4)  # ensure_ascii=False后生成的json文件中就不会包含\u开头的Unicode字符编码

            tsv_file.close()
            json_file.close()
            return os.path.join(os.path.dirname(path), output_file)
        elif args.dev_data.endswith('.json'):
            # 处理json文件
            pass
        else:
            # 不支持的文件类型
            pass


    # dev_data = MyData(tsv2json(args.dev_data))
    dev_data = MyData(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id,
                                                      context_window=args.context_window)  # dev_samples共1338
    dev_batches = batchify(dev_samples, args.eval_batch_size)  # eval_batch_size=32，共42个batch

    if args.do_train:
        # train_data = MyData(tsv2json(args.train_data))
        train_data = MyData(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length,
                                                              ner_label2id=ner_label2id,
                                                              context_window=args.context_window)  # train_samples共1709个（50条数据共1709句话），是由50条数据扩展的，如第一条数据中分为8(子)句。。。
        train_batches = batchify(train_samples, args.train_batch_size)  # train_batch_size=8（每次送入8句话），共214个batch
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                        if 'bert' not in n], 'lr': args.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not (args.bertadam))
        t_total = len(train_batches) * args.num_epoch  # 214*100=21400
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * args.warmup_proportion), t_total)

        # adversarial_training
        if args.adv_training == 'fgm':
            adv = FGM(model=model, param_name='word_embeddings')
        elif args.adv_training == 'pgd':
            adv = PGD(model=model, param_name='word_embeddings')

        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.eval_per_epoch  # 214//1=214
        for _ in tqdm(range(args.num_epoch)):  # 100
            if args.train_shuffle:  # False
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):  # 214
                # output_dict = model.run_batch(train_batches[i], training=True)  # 送入第1个batch
                output_dict, batch_input = model.run_batch(train_batches[i], training=True, label_batch=label_batch,
                                                           args=args)  # 送入第1个batch
                # 上式等号左边的output_dict包括：output_dict['ner_loss'] = ner_loss.sum()，output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
                loss = output_dict['ner_loss']
                loss.backward()

                # 对抗训练
                if args.adv_training:
                    adv.adversarial_training(args, batch_input, optimizer)

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:  # print_loss_step=100，每100步打印
                    logger.info('\n Epoch=%d, iter=%d, loss=%.5f' % (_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:  # eval_step=214，每214验证一次
                    f1 = evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1 * 100))
                        save_model(model, args)

    if args.do_eval:
        args.bert_model_dir = args.output_dir
        # model = EntityModel(args, num_ner_labels=num_ner_labels)
        # rz+
        # model = EntityModel_label(args, num_ner_labels=num_ner_labels)
        model = EntityModel_label_RICON(args)
        # rz+
        if args.eval_test:
            test_data = MyData(tsv2json(args.test_data))
            # test_data = MyData(args.test_data)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        else:
            test_data = MyData(tsv2json(args.dev_data))
            # test_data = MyData(args.dev_data)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id,
                                                            context_window=args.context_window)
        test_batches = batchify(test_samples, args.eval_batch_size)
        evaluate(model, test_batches, test_ner)
        output_ner_predictions(model, test_batches, test_data, output_file=prediction_file)
