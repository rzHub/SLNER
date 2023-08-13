from torch.utils.data import Dataset
import json

class MyData(Dataset):

    def __init__(self, json_file):
        self.js = self._read(json_file)
        # self.documents = [Document(js) for js in self.js]
        # self.sentence =for i in self.js[i]

    def _read(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as js:
            datas = json.load(js)
            result = []
            for data in datas:
                text = data['text']
                # 截断句子长度
                if len(text) > 510:
                    sent_list = [char for char in text]
                    sent_ch = sent_list[:510]
                    sent_ch_tup = tuple(sent_ch)
                    text_ch = sent_ch_tup
                else:
                    sent_list = [char for char in text]
                    sent_tup = tuple(sent_list)
                    text_ch = sent_tup
                one_label = []
                span_label = data['label']
                for label in span_label:  # ('NR', 22, 24)
                    c = label[0]
                    s = label[1]
                    e = label[2]
                    one_label.append((c, s, e))

                result.append({'text': text_ch, 'label': one_label})

        return result

    def __getitem__(self, ix):
        text = self.js[ix]['text']
        label = self.js[ix]['label']
        return text, label

    def __len__(self):
        return len(self.js)
