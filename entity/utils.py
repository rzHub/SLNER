import numpy as np
import json
import logging
from typing import Union, List

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['text']) > 650:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info('单批量样本: %s, \n 长度：%d', samples[i]['text'], len(samples[i]['text']))
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i + batch_size])

    assert (sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

def convert_dataset_to_samples(dataset, max_span_length, ner_label2id=None, context_window=0, split=0):
    """
    Extract sentences and gold entities from a dataset
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0
    
    # if split == 0:
    #     data_range = (0, len(dataset))
    # elif split == 1:
    #     data_range = (0, int(len(dataset)*0.9))
    # elif split == 2:
    #     data_range = (int(len(dataset)*0.9), len(dataset))

    for i, sent in enumerate(dataset):
        # if c < data_range[0] or c >= data_range[1]:
        #     continue
        # for i, sent in enumerate(doc):
        num_ner += len(sent[-1])
        sample = {}
        # if context_window != 0 and len(sent.text) > context_window:
        #     logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
        #     # print('Exclude:', sample)
        #     # continue
        sample['text'] = sent[0]
        sample['sent_length'] = len(sent[0])
        # sent_start = 0
        # sent_end = len(sample['tokens'])
        max_len = max(max_len, len(sent[0]))
        max_ner = max(max_ner, len(sent[-1]))

        # 生成三角mask，维度应为(input_ids[1], input_ids[1]),即句子长度*句子长度
        if max_span_length <= 0:
            triangle_mask = torch.triu(
                torch.ones(len(sent[0]), len(sent[0])), diagonal=0).bool()
        else:
            triangle_mask = (torch.triu(
                torch.ones(len(sent[0]), len(sent[0])), diagonal=0) - torch.triu(
                torch.ones(len(sent[0]), len(sent[0])), diagonal=max_span_length)).bool()  # (104,104)的候选span矩阵，第一行前四位为Ture，
        span_indices = triangle_mask.nonzero()
        span_indices = span_indices.tolist()
        tr_spans = []
        for a_span in span_indices:
            tr_span = (a_span[0], a_span[1], a_span[1]-a_span[0]+1)
            tr_spans.append(tr_span)
        sample['spans'] = tr_spans

        tr_gold_spans = sent[-1]   # [('NR', 22, 24), ('NT', 13, 18), ('NT', 36, 41)]
        gold_spans = [(i[-2], i[-1]) for i in tr_gold_spans]
        spans = [(i[0], i[1]) for i in tr_spans]

        span_label = [0] * len(tr_spans)
        for i, tr_span in enumerate(tr_spans):
            for gold_span in tr_gold_spans:
                label = gold_span[0]
                start = gold_span[1]
                end = gold_span[2]
                if tr_span[0] == start and tr_span[1] == end:
                    span_label[i] = ner_label2id[label]
                    break

        sample['spans_label'] = span_label
        samples.append(sample)

    avg_length = sum([len(sample['text']) for sample in samples]) / len(samples)
    max_length = max([len(sample['text']) for sample in samples])
    logger.info('从数据集提取出 %d 个样例, 共 %d NER labels' % (len(samples), num_ner))
    logger.info('最大样例长度: %d, 最大NER数量: %d' % (max_len, max_ner))
    return samples, num_ner

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

# rz+
import os
import torch
def getlabel_embedding(args, tokenizer):
    label_length = args.label_length
    file_path = os.path.join(args.data_dir, args.ner_label_type)  # args.data_dir='data/ace05/ner_label_type'
    f = open(file_path, "r", encoding='utf-8')
    label_type = []
    for l_idx, line in enumerate(f):
        c = json.loads(line)
        no = c['label_no']
        context = c['label_context']  # str格式
        # context = c['label_context']+' '+c['label_related']  # str格式
        input_ids = tokenizer.encode_plus(context, return_tensors='pt', padding='max_length', max_length=label_length)
        # token = tokenizer.tokenize(context)  # 分为子词，如['犯','罪','嫌'......]
        # target_tokens = [tokenizer.cls_token] + token + [tokenizer.sep_token]    # 前加[cls],后加[sep]
        # L = len(target_tokens)
        # # ①ids编码
        # input_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        # input_ids += [0] * (5 - L)   # 不够5的补0
        # input_ids = torch.tensor(input_ids, dtype=torch.int64)    # 转为tensor
        # # ②位置编码
        # lable_position = list(range(L)) + [0] * (args.max_label_length - L)
        # lable_position = torch.tensor(lable_position, dtype=torch.int64)
        # # ③注意力矩阵(150*150),前64*64为1，其余补零部分为0    ？？？为什么时矩阵？？？--->因为要遮住前面的（见备忘录笔记）
        # label_attention = torch.zeros((args.max_label_length, args.max_label_length), dtype=torch.int64)
        # label_attention[:L,:L] = 1

        label_type.append({'no': no, 'label_input': input_ids['input_ids'],'lable_position':input_ids['token_type_ids'], 'label_attention':input_ids['attention_mask']})
    f.close()
    label_type.sort(key=lambda x: x['no'])   # 输出的标签按序排列

    label_bacth =[torch.stack([x['label_input'].squeeze(0) for x in label_type],dim=0),
                  torch.stack([x['label_attention'].squeeze(0) for x in label_type], dim=0),
                  torch.stack([x['lable_position'].squeeze(0) for x in label_type], dim=0)]   # 转为tensor,从(1,8)先变为(8)再变为(8,8)

    return label_bacth   # (8, 8)8个标签，每个标签编码最长度为8
# rz+


def count_params(model_or_params: Union[torch.nn.Module, torch.nn.Parameter, List[torch.nn.Parameter]],
                 return_trainable=True, verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been iterated ONCE.
    Hence, `model_or_params` passed-in must be a `List` of parameters (which can be iterated multiple times).
    """
    if isinstance(model_or_params, torch.nn.Module):
        model_or_params = list(model_or_params.parameters())
    elif isinstance(model_or_params, torch.nn.Parameter):
        model_or_params = [model_or_params]
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of `torch.nn.Parameter`, "
                        "`model_or_params` should NOT be a `Generator`. ")

    num_trainable = sum(p.numel() for p in model_or_params if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_or_params if not p.requires_grad)

    if verbose:
        logger.info(f"The model has {num_trainable + num_frozen:,} parameters, "
                    f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen.")

    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen
