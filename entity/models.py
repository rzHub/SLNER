import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用的GPU编号，多个GPU可以用逗号分隔，例如"0,1"
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from allennlp.modules import FeedForward
from .layer import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from entity.loss import FocalLoss, LabelSmoothingCrossEntropy
from entity.utils import count_params
import logging

logger = logging.getLogger('root')


# RICON+
class BertForEntity_label_RICON(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150, max_span_length=8, args=None):
        super().__init__(config)

        self.bert = BertModel(config)
        self.label_bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        # self.num_labels = num_ner_labels
        # self.ngram = args.ngram
        self.max_span_length = args.max_span_length

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=args.ricon_hidden_size * 2,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        # activations=F.relu,   # 会报错，改为下面
                        activations=nn.ReLU(),
                        dropout=0.2),
            # nn.Linear(head_hidden_dim, num_ner_labels)
            nn.Linear(head_hidden_dim, config.hidden_size)  # config.hidden_size=768???
        )

        # RICON模块+
        hidden_size = args.ricon_hidden_size  # 200
        # Regularity-aware module
        self.regularity_aware_lstm = nn.LSTM(  # (768,200,...)
            input_size=config.hidden_size,  # 768
            hidden_size=hidden_size,  # 200
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4, )
        # self.reg = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.self_attentive_span_extractor = SelfAttentiveSpanExtractor(
            2 * hidden_size)  # 通过（线性自注意力）的span表示
        self.endpoint_span_extractor = EndpointSpanExtractor(
            2 * hidden_size, combination=args.combination)  # 通过（头cat尾）的span表示
        # self.reg_linear = nn.Linear(2 * hidden_size, 1)
        # self.aware_biaffine = Biaffine(
        #     2 * hidden_size, 2 * hidden_size, bias_x=False, bias_y=False
        # )
        self.u2 = nn.Linear(  # u2:(800,400)
            self.endpoint_span_extractor.get_output_dim(),
            2 * hidden_size)
        self.u3 = nn.Linear(4 * hidden_size, 1)  # u3:(800,1)
        # self.type_linear = nn.Linear(2 * hidden_size, self.num_labels)  # (400,4)   去掉输出分类器头，这样模型可以迁移到不同数据集，以实现few-shot
        # RICON模块+

        self.init_weights()

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        # sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)   # 报错，需要加一个return_dict=False，仍为tensor而不是str
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask, return_dict=False)

        sequence_output = self.hidden_dropout(sequence_output)

        # (1) Regularity-aware module 规则感知模块
        h_aware = self.regularity_aware_lstm(sequence_output)[0]  # bilstm输出：(8,104,400)为什么是400？？？因为是双向lstm
        # bs, seqlen = h_aware.shape[0], h_aware.shape[1]
        # # seqlen = input_ids.shape
        # # 生成三角mask，维度应为(input_ids[1], input_ids[1]),即句子长度*句子长度
        # if self.max_span_length <= 0:
        #     triangle_mask = torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=0).bool()
        # else:
        #     triangle_mask = (torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=0) - torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=self.max_span_length)).bool()  # (104,104)的候选span矩阵，第一行前四位为Ture，
        # triangle_mask = triangle_mask.cuda()                                        # 第二行相当于将第一行右移一位，以此类推
        # RICON+
        # h_reg (equation 5-7)  线性注意力
        # h_reg shape [bs, seqlen, seqlen, 2d]
        # a_aware = self.reg_linear(h_aware)
        # h_reg = []
        # for i in range(seqlen):
        #     for j in range(seqlen):
        #         if i == j:
        #             h_reg.append(h_aware[:, i])
        #         elif j > i:
        #             if j - i + 1 > self.ngram:
        #                 continue
        #             reg_at = a_aware[:, i : j + 1].softmax(1)
        #             h_reg.append((reg_at * h_aware[:, i : j + 1]).sum(1))
        # h_reg = torch.stack(h_reg, dim=1)
        # span_indices = triangle_mask.nonzero().unsqueeze(0).expand(
        #     bs, -1, -1)  # span索引:(8,410,2),410个候选span，格式如(0,0),(0,1),(0,2),(0,3),...(1,1),(1,2),...
        span_indices = spans[:, :, 0:2]
        h_reg = self.self_attentive_span_extractor(
            h_aware,
            span_indices=span_indices)  # 规则特征(8,410,400)用自注意力span表征实现，我觉得这里输出维度不应为400，应该为hidden_size*span宽度？？？(但问题是span宽度由1-4不等)

        # h_span (equation 8)
        # h_span shape [bs, seqlen, seqlen, 2d]
        # 不加这个
        # self.aware_biaffine(h_aware, h_aware)
        h_span = self.u2(
            self.endpoint_span_extractor(
                h_aware, span_indices)).reshape_as(h_reg)  # (8,410,400)[bs, seqlen, seqlen, 2d]
        # h_span = self.u2(
        #     torch.cat(
        #         [
        #             h_aware[:, None, :, :].expand(-1, seqlen, -1, -1),
        #             h_aware[:, :, None, :].expand(-1, -1, seqlen, -1),
        #         ],
        #         dim=-1,
        #     )
        # )
        # # [bs, ll, 2d]
        # h_span = h_span.masked_select(triangle_mask[None, :, :, None]).reshape(
        #     bs, -1, hidden_size_2
        # )
        # h_sij  (equation 9-10)
        g_sij = torch.sigmoid(
            self.u3(torch.cat([h_span, h_reg], dim=-1)))  # (8,410,1)
        h_sij = g_sij * h_span + (1 - g_sij) * h_reg  # (8,410,400)
        spans_embedding = h_sij
        # aware_output (equation 11)
        # aware_output = self.type_linear(h_sij)  # (8,410,4)

        # """
        # spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        # spans_mask: (batch_size, num_spans, )
        # """
        # spans_start = spans[:, :, 0].view(spans.size(0), -1)
        # spans_start_embedding = batched_index_select(sequence_output, spans_start)
        # spans_end = spans[:, :, 1].view(spans.size(0), -1)
        # spans_end_embedding = batched_index_select(sequence_output, spans_end)
        #
        # spans_width = spans[:, :, 2].view(spans.size(0), -1)
        # spans_width_embedding = self.width_embedding(spans_width)
        #
        # # Concatenate embeddings of left/right points and the width embedding
        # spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        # """
        # spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        # """
        return spans_embedding

    def _get_label_embeddings(self, label_input, label_attention, label_position):
        sequence_output_label, pooled_output_label = self.label_bert(input_ids=label_input,
                                                                     token_type_ids=label_position,
                                                                     attention_mask=label_attention, return_dict=False)
        pooled_output_label = self.hidden_dropout(pooled_output_label)
        label_embedding = pooled_output_label
        # # 标签不使用[CLS],改为和token一样的编码模式
        # label_embedding = sequence_output_label
        # h_aware_label = self.regularity_aware_lstm(sequence_output_label)[0]
        # span_indices = spans[:, :, 0:2]
        # h_reg = self.self_attentive_span_extractor(
        #     h_aware_label, span_indices=span_indices)
        # h_span = self.u2(
        #     self.endpoint_span_extractor(
        #         h_aware_label, span_indices)).reshape_as(h_reg)
        # g_sij = torch.sigmoid(
        #     self.u3(torch.cat([h_span, h_reg], dim=-1)))   # (8,410,1)
        # h_sij = g_sij * h_span + (1 - g_sij) * h_reg       # (8,410,400)
        # label_embedding = h_sij
        # # 标签不使用[CLS],改为和token一样的编码模式

        return label_embedding

    def forward(self, input_ids, spans, spans_mask, label_input, label_attention, label_position, spans_ner_label=None,
                token_type_ids=None, attention_mask=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)  # (8,82,1686)(batch_size, num_spans, hidden_size*2+embedding_dim)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]  # bio(8,82,768)82个token改为span(8,1544,768)1544个span
        label_embedding = self._get_label_embeddings(label_input, label_attention,
                                                     label_position)  # (8,768)(number_labels, hidden_size)
        # label_embedding = label_embedding.detach()   # 保存不更新
        tag_lens, hidden_size = label_embedding.shape
        label_embedding = label_embedding.expand(spans_embedding.shape[0], tag_lens, hidden_size)  # (8,8,768)
        label_embedding = label_embedding.transpose(2, 1)  # (8,768,8)
        matrix_embeddings = torch.matmul(logits, label_embedding)  # (8,1544,8)
        softmax_embedding = nn.Softmax(dim=-1)(matrix_embeddings)  # 对点乘结果进行打分
        label_indexs = torch.argmax(softmax_embedding, dim=-1)  # (8,332)

        if spans_ner_label is not None:  # 训练阶段  (8,82),表示82个span,现在要改为(8,410)
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:  # (8,104)
                active_loss = spans_mask.view(-1) == 1
                # active_logits = logits.view(-1, logits.shape[-1])
                active_logits = matrix_embeddings.view(-1, matrix_embeddings.shape[-1])  # (656,8)(12352,8??)
                ##### active_labels_indexs = label_indexs.view(-1)
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )  # (656)(1184??)
                loss = loss_fct(active_logits, active_labels)
                ##### loss1 = loss_fct(active_labels_indexs.float(), active_labels)
            else:
                # loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
                loss = loss_fct(matrix_embeddings.view(-1, matrix_embeddings.shape[-1]), spans_ner_label.view(-1))
            # return loss, logits, spans_embedding
            return loss, matrix_embeddings, spans_embedding
        else:
            # return logits, spans_embedding, spans_embedding
            return matrix_embeddings, spans_embedding, spans_embedding

class EntityModel_label_RICON():

    def __init__(self, args):
        super().__init__()

        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        if args.use_albert:
            pass
            # self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            # self.bert_model = AlbertForEntity_label.from_pretrained(bert_model_name,
            #                                                         max_span_length=args.max_span_length)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = BertForEntity_label_RICON.from_pretrained(bert_model_name,
                                                                        max_span_length=args.max_span_length,
                                                                        args=args)  # 去掉了参数num_ner_labels
            count_params(self.bert_model)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        # if torch.cuda.device_count() > 1:
        #     self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        sentence = samples_list[0]
        sentence = samples_list[0]
        for sample in samples_list:
            text = sample['text']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']

            # 截断句子长度
            if len(text) > 510:
                text = text[:510]

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(text, spans,
                                                                                               spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list,
                                                                            spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        # logger.info(final_tokens_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_bert_spans_tensor)
        # logger.info(final_bert_spans_tensor.shape)
        # logger.info(final_spans_mask_tensor.shape)
        # logger.info(final_spans_ner_label_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def run_batch(self, samples_list, try_cuda=True, training=True, label_batch=None, args=None):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list, training)
        label_input, label_attention, label_position = label_batch[0], label_batch[1], label_batch[2]
        batch_input = {
            'input_ids': tokens_tensor.to(self._model_device),
            'spans': bert_spans_tensor.to(self._model_device),
            'spans_mask': spans_mask_tensor.to(self._model_device),
            'spans_ner_label': spans_ner_label_tensor.to(self._model_device),
            'attention_mask': attention_mask_tensor.to(self._model_device),
            'label_input': label_input.to(self._model_device),
            'label_attention': label_attention.to(self._model_device),
            'label_position': label_position.to(self._model_device)
        }
        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(  # loss, matrix_embeddings, spans_embedding
                input_ids=tokens_tensor.to(self._model_device),
                spans=bert_spans_tensor.to(self._model_device),
                spans_mask=spans_mask_tensor.to(self._model_device),
                spans_ner_label=spans_ner_label_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
                label_input=label_input.to(self._model_device),
                label_attention=label_attention.to(self._model_device),
                label_position=label_position.to(self._model_device)
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
            # softmax_embedding = nn.Softmax(dim=-1)(ner_logits)  # 对点乘结果进行打分
            # label_indexs = torch.argmax(softmax_embedding, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    # matrix_embeddings, spans_embedding, spans_embedding
                    input_ids=tokens_tensor.to(self._model_device),
                    spans=bert_spans_tensor.to(self._model_device),
                    spans_mask=spans_mask_tensor.to(self._model_device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._model_device),
                    label_input=label_input.to(self._model_device),
                    label_attention=label_attention.to(self._model_device),
                    label_position=label_position.to(self._model_device)
                )
            _, predicted_label = ner_logits.max(2)  # 在第2维度上取最大值 ner_logits的维度为(8,1544,8)，即取8类中的最大概率
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict, batch_input
# RICON+



# fewshot+
# class SelfAttentionLayer(nn.Module):
#     def __init__(self, input_size, num_heads):
#         super(SelfAttentionLayer, self).__init__()
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.attention_weights = nn.Linear(input_size, num_heads)
#
#     def forward(self, spans_embedding):
#         attention_weights = self.attention_weights(spans_embedding)
#         attention_scores = torch.softmax(attention_weights, dim=1)
#         attended_spans = torch.matmul(attention_scores.transpose(1, 2), spans_embedding)
#         return attended_spans


class BertForEntity_label_RICON_fs(BertPreTrainedModel):
    def __init__(self, config, head_hidden_dim=150, width_embedding_dim=150, max_span_length=8, args=None):
        super().__init__(config)

        self.bert = BertModel(config)
        self.label_bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        # self.num_labels = num_ner_labels
        # self.ngram = args.ngram
        self.max_span_length = args.max_span_length

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=args.ricon_hidden_size * 2,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        # activations=F.relu,   # 会报错，改为下面
                        activations=nn.ReLU(),
                        dropout=0.2),
            # nn.Linear(head_hidden_dim, num_ner_labels)
            nn.Linear(head_hidden_dim, config.hidden_size)  # config.hidden_size=768???
        )

        # RICON模块+
        hidden_size = args.ricon_hidden_size  # 200
        # Regularity-aware module
        self.regularity_aware_lstm = nn.LSTM(  # (768,200,...)
            input_size=config.hidden_size,  # 768
            hidden_size=hidden_size,  # 200
            num_layers=1,
            bidirectional=True,  # 双向LSTM
            batch_first=True,
            dropout=0.4, )
        # self.reg = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.self_attentive_span_extractor = SelfAttentiveSpanExtractor(
            2 * hidden_size)  # 通过（线性自注意力）的span表示
        self.endpoint_span_extractor = EndpointSpanExtractor(
            2 * hidden_size, combination=args.combination)  # 通过（头cat尾）的span表示
        # self.reg_linear = nn.Linear(2 * hidden_size, 1)
        # self.aware_biaffine = Biaffine(
        #     2 * hidden_size, 2 * hidden_size, bias_x=False, bias_y=False
        # )
        self.u2 = nn.Linear(  # u2:(800,400)
            self.endpoint_span_extractor.get_output_dim(),
            2 * hidden_size)
        self.u3 = nn.Linear(4 * hidden_size, 1)  # u3:(800,1)
        # self.type_linear = nn.Linear(2 * hidden_size, self.num_labels)  # (400,4)
        # RICON模块+

        self.loss_type = args.loss_type
        self.init_weights()

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        # sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)   # 报错，需要加一个return_dict=False，仍为tensor而不是str
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask, return_dict=False)

        sequence_output = self.hidden_dropout(sequence_output)

        # (1) Regularity-aware module 规则感知模块
        h_aware = self.regularity_aware_lstm(sequence_output)[0]  # bilstm输出：(8,104,400)为什么是400？？？因为是双向lstm
        # bs, seqlen = h_aware.shape[0], h_aware.shape[1]
        # # seqlen = input_ids.shape
        # # 生成三角mask，维度应为(input_ids[1], input_ids[1]),即句子长度*句子长度
        # if self.max_span_length <= 0:
        #     triangle_mask = torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=0).bool()
        # else:
        #     triangle_mask = (torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=0) - torch.triu(
        #         torch.ones(seqlen, seqlen), diagonal=self.max_span_length)).bool()  # (104,104)的候选span矩阵，第一行前四位为Ture，
        # triangle_mask = triangle_mask.cuda()                                        # 第二行相当于将第一行右移一位，以此类推
        # RICON+
        # h_reg (equation 5-7)  线性注意力
        # h_reg shape [bs, seqlen, seqlen, 2d]
        # a_aware = self.reg_linear(h_aware)
        # h_reg = []
        # for i in range(seqlen):
        #     for j in range(seqlen):
        #         if i == j:
        #             h_reg.append(h_aware[:, i])
        #         elif j > i:
        #             if j - i + 1 > self.ngram:
        #                 continue
        #             reg_at = a_aware[:, i : j + 1].softmax(1)
        #             h_reg.append((reg_at * h_aware[:, i : j + 1]).sum(1))
        # h_reg = torch.stack(h_reg, dim=1)
        # span_indices = triangle_mask.nonzero().unsqueeze(0).expand(
        #     bs, -1, -1)  # span索引:(8,410,2),410个候选span，格式如(0,0),(0,1),(0,2),(0,3),...(1,1),(1,2),...
        span_indices = spans[:, :, 0:2]
        h_reg = self.self_attentive_span_extractor(
            h_aware,
            span_indices=span_indices)  # 规则特征(8,410,400)用自注意力span表征实现，我觉得这里输出维度不应为400，应该为hidden_size*span宽度？？？(但问题是span宽度由1-4不等)

        # h_span (equation 8)
        # h_span shape [bs, seqlen, seqlen, 2d]
        # 不加这个
        # self.aware_biaffine(h_aware, h_aware)
        h_span = self.u2(
            self.endpoint_span_extractor(
                h_aware, span_indices)).reshape_as(h_reg)  # (8,410,400)[bs, seqlen, seqlen, 2d]
        # h_span = self.u2(
        #     torch.cat(
        #         [
        #             h_aware[:, None, :, :].expand(-1, seqlen, -1, -1),
        #             h_aware[:, :, None, :].expand(-1, -1, seqlen, -1),
        #         ],
        #         dim=-1,
        #     )
        # )
        # # [bs, ll, 2d]
        # h_span = h_span.masked_select(triangle_mask[None, :, :, None]).reshape(
        #     bs, -1, hidden_size_2
        # )
        # h_sij  (equation 9-10)
        g_sij = torch.sigmoid(
            self.u3(torch.cat([h_span, h_reg], dim=-1)))  # (8,410,1)
        h_sij = g_sij * h_span + (1 - g_sij) * h_reg  # (8,410,400)
        spans_embedding = h_sij
        # aware_output (equation 11)
        # aware_output = self.type_linear(h_sij)  # (8,410,4)

        # """
        # spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        # spans_mask: (batch_size, num_spans, )
        # """
        # spans_start = spans[:, :, 0].view(spans.size(0), -1)
        # spans_start_embedding = batched_index_select(sequence_output, spans_start)
        # spans_end = spans[:, :, 1].view(spans.size(0), -1)
        # spans_end_embedding = batched_index_select(sequence_output, spans_end)
        #
        # spans_width = spans[:, :, 2].view(spans.size(0), -1)
        # spans_width_embedding = self.width_embedding(spans_width)
        #
        # # Concatenate embeddings of left/right points and the width embedding
        # spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        # """
        # spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        # """
        return spans_embedding

    def _get_label_embeddings(self, label_input, label_attention, label_position):
        sequence_output_label, pooled_output_label = self.label_bert(input_ids=label_input,
                                                                     token_type_ids=label_position,
                                                                     attention_mask=label_attention, return_dict=False)
        # # 计算注意力权重
        # attention_scores = torch.matmul(sequence_output_label, sequence_output_label.transpose(1, 2))  # 内积计算注意力得分
        # attention_weights = nn.functional.softmax(attention_scores, dim=1)  # 注意力权重归一化
        #
        # # 应用注意力权重到label编码上
        # weighted_label_embedding = torch.matmul(attention_weights, sequence_output_label)
        #
        # # 获得句子层面的全局label编码
        # global_label_encoding = weighted_label_embedding  # global_span_encoding即为句子层面的全局span编码，维度为(batch_size, span个数, hidden_dim)
        # global_label_embedding = global_label_encoding[0, start_index:end_index + 1, :]
        pooled_output_label = self.hidden_dropout(pooled_output_label)
        label_embedding = pooled_output_label
        return label_embedding

    def forward(self, input_ids, spans, spans_mask, label_input, label_attention, label_position, spans_ner_label=None,
                token_type_ids=None, attention_mask=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)  # (8,82,1686)(batch_size, num_spans, hidden_size*2+embedding_dim)

        # # 计算注意力权重
        # attention_scores = torch.matmul(spans_embedding, spans_embedding.transpose(1, 2))  # 内积计算注意力得分
        # attention_weights = nn.functional.softmax(attention_scores, dim=1)  # 注意力权重归一化
        #
        # # 应用注意力权重到span编码上
        # weighted_spans_embedding = torch.matmul(attention_weights, spans_embedding)
        #
        # # 获得句子层面的全局span编码
        # global_span_encoding = weighted_spans_embedding  # global_span_encoding即为句子层面的全局span编码，维度为(batch_size, span个数, hidden_dim)
        # spans_embedding = global_span_encoding
        # spans_embedding = self.hidden_dropout(spans_embedding)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]  # bio(8,82,768)82个token改为span(8,1544,768)1544个span
        label_embedding = self._get_label_embeddings(label_input, label_attention,
                                                     label_position)  # (8,768)(number_labels, hidden_size)
        # label_embedding = label_embedding.detach()   # 保存不更新
        tag_lens, hidden_size = label_embedding.shape
        label_embedding = label_embedding.expand(spans_embedding.shape[0], tag_lens, hidden_size)  # (8,8,768)
        label_embedding = label_embedding.transpose(2, 1)  # (8,768,8)
        matrix_embeddings = torch.matmul(logits, label_embedding)  # (8,1544,8)
        # matrix_embeddings = self.hidden_dropout(matrix_embeddings)
        # softmax_embedding = nn.Softmax(dim=-1)(matrix_embeddings)           # 对点乘结果进行打分
        # label_indexs = torch.argmax(softmax_embedding,dim=-1)

        if spans_ner_label is not None:  # 训练阶段  (8,82),表示82个span,现在要改为(8,410)
            assert self.loss_type in ["lsce", "focal", "ce"]
            if self.loss_type == "lsce":
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == "focal":
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:  # (8,104)
                active_loss = spans_mask.view(-1) == 1
                # active_logits = logits.view(-1, logits.shape[-1])
                active_logits = matrix_embeddings.view(-1, matrix_embeddings.shape[-1])  # (656,8)(12352,8??)
                # active_logits = softmax_embedding.view(-1, softmax_embedding.shape[-1])  # (656,8)(12352,8??)
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )  # (656)(1184??)
                loss = loss_fct(active_logits, active_labels)
            else:
                # loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
                loss = loss_fct(matrix_embeddings.view(-1, matrix_embeddings.shape[-1]), spans_ner_label.view(-1))
                # loss = loss_fct(softmax_embedding.view(-1, softmax_embedding.shape[-1]), spans_ner_label.view(-1))
            # return loss, logits, spans_embedding
            return loss, matrix_embeddings, spans_embedding
        else:
            # return logits, spans_embedding, spans_embedding
            return matrix_embeddings, spans_embedding, spans_embedding

class EntityModel_label_RICON_fs():

    def __init__(self, args):
        super().__init__()

        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        if args.use_albert:
            pass
            # self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            # self.bert_model = AlbertForEntity_label.from_pretrained(bert_model_name,
            #                                                         max_span_length=args.max_span_length)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            if args.eval_test:
                pretrain_path = 'out/entity_fewshot/pytorch_model.bin'
            else:
            # pretrain_path = 'out/RHD(x,y,xy)/pytorch_model.bin'
            # pretrain_path = 'out/RHD_yi_75.25/pytorch_model.bin'               # 双塔（伪孪生网络）
                pretrain_path = 'out/HRD13.59/pytorch_model.bin'  # 双塔（伪孪生网络）
            # pretrain_path = 'out/HRD-IS/pytorch_model.bin'  # without inner span
            # pretrain_path = 'out/singlebert_news86.72/pytorch_model.bin'  # 单塔（孪生网络）
            self.bert_model = BertForEntity_label_RICON_fs.from_pretrained(bert_model_name,
                                                                           max_span_length=args.max_span_length,
                                                                           args=args)
            self.bert_model.load_state_dict(torch.load(pretrain_path), strict=False)  # 加载源域训练好的模型
            count_params(self.bert_model)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        # if torch.cuda.device_count() > 1:
        #     self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        sentence = samples_list[0]
        sentence = samples_list[0]
        for sample in samples_list:
            text = sample['text']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']

            # 截断句子长度
            if len(text) > 510:
                text = text[:510]

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(text, spans,
                                                                                               spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list,
                                                                            spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        # logger.info(final_tokens_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_bert_spans_tensor)
        # logger.info(final_bert_spans_tensor.shape)
        # logger.info(final_spans_mask_tensor.shape)
        # logger.info(final_spans_ner_label_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def run_batch(self, samples_list, try_cuda=True, training=True, label_batch=None, args=None):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list, training)
        label_input, label_attention, label_position = label_batch[0], label_batch[1], label_batch[2]
        batch_input = {
            'input_ids': tokens_tensor.to(self._model_device),
            'spans': bert_spans_tensor.to(self._model_device),
            'spans_mask': spans_mask_tensor.to(self._model_device),
            'spans_ner_label': spans_ner_label_tensor.to(self._model_device),
            'attention_mask': attention_mask_tensor.to(self._model_device),
            'label_input': label_input.to(self._model_device),
            'label_attention': label_attention.to(self._model_device),
            'label_position': label_position.to(self._model_device)
        }
        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                # 等号左边三个值分别对应loss, matrix_embeddings, spans_embedding
                input_ids=tokens_tensor.to(self._model_device),
                spans=bert_spans_tensor.to(self._model_device),
                spans_mask=spans_mask_tensor.to(self._model_device),
                spans_ner_label=spans_ner_label_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
                label_input=label_input.to(self._model_device),
                label_attention=label_attention.to(self._model_device),
                label_position=label_position.to(self._model_device)
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
            # softmax_embedding = nn.Softmax(dim=-1)(ner_logits)  # 对点乘结果进行打分
            # label_indexs = torch.argmax(softmax_embedding, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    # 等号左边三个值分别对应matrix_embeddings, spans_embedding, spans_embedding
                    input_ids=tokens_tensor.to(self._model_device),
                    spans=bert_spans_tensor.to(self._model_device),
                    spans_mask=spans_mask_tensor.to(self._model_device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._model_device),
                    label_input=label_input.to(self._model_device),
                    label_attention=label_attention.to(self._model_device),
                    label_position=label_position.to(self._model_device)
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict, batch_input
# fewshot+
