task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'msra': ['NS', 'NR', 'NT'],
    'onto4': ['LOC', 'PER', 'GPE', 'ORG'],
    'FOOD': ['FRU', 'SYM', 'MER', 'ECP', 'SIG', 'CAU'],
    'weibo': ['PER.NOM', 'PER.NAM', 'LOC.NOM', 'LOC.NAM', 'GPE.NOM', 'GPE.NAM', 'ORG.NOM', 'ORG.NAM'],
    'RISK': ['LOC-PROV', 'LOC-PREF', 'LOC-COUNT', 'FOOD-VEG', 'FOOD-FRUIT',
             'FOOD-MEAT', 'FOOD-GRAIN', 'FOOD-DAIRY', 'CO-PROC', 'CO-PROD',
             'CO-TRAD', 'CO-CATE', 'CO-MATE', 'ORG-REGU', 'ORG-ADMI',
             'ORG-LEGA', 'ORG-SOCI', 'RISK-HIGH', 'RISK-MID', 'RISK-LOW'],
    'resume': ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE'],
    'RMRB': ['LOC', 'ORG', 'PER'],
    'cluener': ['ADDRESS', 'BOOK', 'COMPANY', 'GAME', 'GOVERNMENT',
                'MOVIE', 'NAME', 'ORGANIZATION', 'POSITION', 'SCENE'],
    'ALL': ['LOC', 'PER', 'GPE', 'ORG', 'CONT', 'EDU', 'PRO',
            'RACE', 'TITLE', 'BOOK', 'GAME', 'GOVERNMENT', 'MOVIE'],
    'newall': ['PER.NOM', 'LOC.NOM', 'ORG.NOM', 'GPE.NOM', 'SCENE', 'MISC', 'XH',
               'HPPX', 'HCCX', 'Time', 'Metric', 'LOC', 'PER',
               'GPE', 'ORG', 'CONT', 'EDU', 'PRO', 'RACE',
               'TITLE', 'Abstract', 'GAME', 'Thing', 'MOVIE'],
    'news': ['LOC', 'PER', 'GPE', 'ORG', 'SCENE',
             'TITLE', 'Abstract', 'GAME', 'MOVIE'],
    'HRD': ['ORG', 'LOC', 'PER', 'Time', 'Thing', 'Metric', 'Abstract', 'Physical', 'Term', 'company',
            'name', 'game', 'movie', 'position', 'address', 'government', 'scene', 'book'],
    'RMRB2014': ['T', 'LOC', 'ORG', 'PER'],
    'rd': ['sym', 'dru', 'dis', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep', 'MISC', 'XH', 'HCCX', 'ORG', 'LOC', 'PER', 'Time', 'Thing', 'Metric', 'Abstract',
            'Physical', 'Product', 'COMPANY', 'NAME', 'GAME', 'MOVIE', 'POSITION', 'ADDRESS',
           'GOVERNMENT', 'SCENE', 'BOOK'],
}

task_fewshot_ner_labels = {
    'msra': ['NS', 'NR', 'NT'],
    'onto4': ['LOC', 'PER', 'GPE', 'ORG'],
    'FOOD': ['FRU', 'SYM', 'MER', 'ECP', 'SIG', 'CAU'],
    'weibo': ['PER.NOM', 'PER.NAM', 'LOC.NOM', 'LOC.NAM', 'GPE.NOM', 'GPE.NAM', 'ORG.NOM', 'ORG.NAM'],
    'RISK': ['LOC-PROV', 'LOC-PREF', 'LOC-COUNT', 'FOOD-VEG', 'FOOD-FRUIT',
             'FOOD-MEAT', 'FOOD-GRAIN', 'FOOD-DAIRY', 'CO-PROC', 'CO-PROD',
             'CO-TRAD', 'CO-CATE', 'CO-MATE', 'ORG-REGU', 'ORG-ADMI',
             'ORG-LEGA', 'ORG-SOCI', 'RISK-HIGH', 'RISK-MID', 'RISK-LOW'],
    'resume': ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE'],
    'RMRB': ['LOC', 'ORG', 'PER'],
    'cluener': ['ADDRESS', 'BOOK', 'COMPANY', 'GAME', 'GOVERNMENT',
                'MOVIE', 'NAME', 'ORGANIZATION', 'POSITION', 'SCENE'],
    'RISK_CU': ['LOC', 'FOOD', 'CO', 'ORG', 'RISK'],
}

# rz+
task_ner_labels_new = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}
# rz+


task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
