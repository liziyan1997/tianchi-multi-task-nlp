#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:46:03 2020

@author: luokai
"""


import json
from collections import defaultdict
from math import log

def split_dataset(dev_data_cnt=5000):
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        cnt = 0
        with open('./tianchi_datasets/' + e + '/total.csv') as f:
            with open('./tianchi_datasets/' + e + '/train.csv', 'w') as f_train:
                with open('./tianchi_datasets/' + e + '/dev.csv', 'w') as f_dev:
                    for line in f:
                        cnt += 1
                        if cnt <= dev_data_cnt[e]:
                            f_dev.write(line)
                        else:
                            f_train.write(line)
                            
def print_one_data(path, name, print_content=False):
    data_cnt = 0
    with open(path) as f:
        for line in f:
            tmp = json.loads(line)
            for _, v in tmp.items():
                data_cnt += 1
                if print_content:
                    print(v)
    print(name, 'contains:', data_cnt, 'numbers of data')

def generate_data():
    label_set = dict()
    label_cnt_set = dict()      # 计算每个任务数据集中每个label对应的数据数量
    dataset_cnt = defaultdict(int)
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        label_set[e] = set()
        label_cnt_set[e] = defaultdict(int)    
        with open('./tianchi_datasets/' + e + '/total.csv') as f:
            for line in f:
                dataset_cnt[e] += 1
                label = line.strip().split('\t')[-1]
                label_set[e].add(label)
                label_cnt_set[e][label] += 1
    print(dataset_cnt)
    max_dataset_cnt = max(dataset_cnt.values())
    for k in label_set:
        label_set[k] = sorted(list(label_set[k]))
    for k, v in label_set.items():
        print(k, v)
    with open('./tianchi_datasets/label.json', 'w') as fw:
        fw.write(json.dumps(label_set))
    label_weight_set = dict() # 按照不同label数据量的多少对label进行赋权，为train阶段weighted loss使用
    label_weight_norm_set = dict()
    for k in label_set:
        label_weight_set[k] = [label_cnt_set[k][e] for e in label_set[k]]
        total_weight = sum(label_weight_set[k])
        label_weight_set[k] = [log(total_weight / e) for e in label_weight_set[k]] # original 
        total_weight = sum(label_weight_set[k])
        label_weight_norm_set[k] = [e*len(label_set[k])*(max_dataset_cnt/dataset_cnt[k])/total_weight for e in label_weight_set[k]]

    # for k, v in label_weight_set.items():
    #     print(k, v)
    print(label_cnt_set)
    with open('./tianchi_datasets/label_weights.json', 'w') as fw:
        fw.write(json.dumps(label_weight_set))
    with open('./tianchi_datasets/label_weights_norm.json', 'w') as fw:
        fw.write(json.dumps(label_weight_norm_set))
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        for name in ['dev', 'train','total']:
            with open('./tianchi_datasets/' + e + '/' + name + '.csv') as fr:
                with open('./tianchi_datasets/' + e + '/' + name + '.json', 'w') as fw:
                    json_dict = dict()
                    for line in fr:
                        tmp_list = line.strip().split('\t')
                        json_dict[tmp_list[0]] = dict()
                        json_dict[tmp_list[0]]['s1'] = tmp_list[1]
                        if e == 'OCNLI':
                            json_dict[tmp_list[0]]['s2'] = tmp_list[2]
                            json_dict[tmp_list[0]]['label'] = tmp_list[3]
                        else:
                            json_dict[tmp_list[0]]['label'] = tmp_list[2]
                    fw.write(json.dumps(json_dict))
    
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        for name in ['dev', 'train']:
            cur_path = './tianchi_datasets/' + e + '/' + name + '.json'
            data_name = e + '_' + name
            print_one_data(cur_path, data_name)
            
    print_one_data('./tianchi_datasets/label.json', 'label_set')
    
if __name__ == '__main__':
    print('-------------------------------start-----------------------------------')
    dev_data_cnt = {
        'TNEWS': 3360,
        'OCNLI': 3387,
        'OCEMOTION': 2694
    }
    split_dataset(dev_data_cnt=dev_data_cnt)
    generate_data()
    print('-------------------------------finish-----------------------------------')