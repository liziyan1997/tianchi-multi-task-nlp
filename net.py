#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:20:35 2020

@author: luokai
"""

import torch
from torch import nn
from transformers import BertModel

NUM_EMB = 1024 # 1024 for roberta_large, 768 for bert

class Net(nn.Module):
    def __init__(self, bert_model):
        super(Net, self).__init__()
        self.bert = bert_model
        self.atten_layer = nn.Linear(NUM_EMB, 16)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.OCNLI_layer = nn.Linear(NUM_EMB, 16 * 3)
        self.OCEMOTION_layer = nn.Linear(NUM_EMB, 16 * 7)
        self.TNEWS_layer = nn.Linear(NUM_EMB, 16 * 15)

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        # print(input_ids, ocnli_ids, ocemotion_ids, tnews_ids)
        if ocnli_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocnli_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocnli_value = self.OCNLI_layer(cls_emb[ocnli_ids, :]).contiguous().view(-1, 16, 3)
            ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocemotion_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocemotion_value = self.OCEMOTION_layer(cls_emb[ocemotion_ids, :]).contiguous().view(-1, 16, 7)
            ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[tnews_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            tnews_value = self.TNEWS_layer(cls_emb[tnews_ids, :]).contiguous().view(-1, 16, 15)
            tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)
        else:
            tnews_out = None
        # print(ocnli_out.size(),ocemotion_out.size(),tnews_out.size())
        return ocnli_out, ocemotion_out, tnews_out

class Net_1(nn.Module):
    def __init__(self, bert_model):
        super(Net_1, self).__init__()
        self.bert = bert_model
        self.atten_layer = nn.Linear(NUM_EMB, 16)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        
        # self.denselayer = nn.Linear(NUM_EMB,256)
        self.OCNLI_layer1 = nn.Linear(NUM_EMB, 256)
        self.OCNLI_layer2 = nn.Linear(256, 3)
        self.OCEMOTION_layer1 = nn.Linear(NUM_EMB, 256)
        self.OCEMOTION_layer2 = nn.Linear(256, 7)
        self.TNEWS_layer1 = nn.Linear(NUM_EMB, 256)
        self.TNEWS_layer2 = nn.Linear(256, 15)

        self.relu = nn.ReLU()

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        # print(input_ids, ocnli_ids, ocemotion_ids, tnews_ids)
        if ocnli_ids.size()[0] > 0:
            ocnli_out = self.relu(self.OCNLI_layer1(cls_emb[ocnli_ids, :]))
            ocnli_out = self.dropout(ocnli_out)
            # ocnli_out = self.relu(self.OCNLI_layer1(ocnli_out))
            ocnli_out = self.OCNLI_layer2(ocnli_out)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            ocemotion_out = self.relu(self.OCEMOTION_layer1(cls_emb[ocemotion_ids, :]))
            ocemotion_out = self.dropout(ocemotion_out)
            # ocemotion_out = self.relu(self.OCEMOTION_layer1(ocemotion_out))
            ocemotion_out = self.OCEMOTION_layer2(ocemotion_out)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            tnews_out = self.relu(self.TNEWS_layer1(cls_emb[tnews_ids, :]))
            tnews_out = self.dropout(tnews_out)
            # tnews_out = self.relu(self.TNEWS_layer1(tnews_out))
            tnews_out = self.TNEWS_layer2(tnews_out)
        else:
            tnews_out = None
        # print(ocnli_out.size(),ocemotion_out.size(),tnews_out.size())
        return ocnli_out, ocemotion_out, tnews_out

class Net_2(nn.Module):
    def __init__(self, bert_model):
        super(Net_2, self).__init__()
        self.bert = bert_model
        # self.atten_layer = nn.Linear(NUM_EMB, 16)
        # self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        
        self.denselayer = nn.Linear(NUM_EMB,256)
        # self.OCNLI_layer1 = nn.Linear(NUM_EMB, 256)
        self.OCNLI_layer2 = nn.Linear(256, 3)
        # self.OCEMOTION_layer1 = nn.Linear(NUM_EMB, 256)
        self.OCEMOTION_layer2 = nn.Linear(256, 7)
        # self.TNEWS_layer1 = nn.Linear(NUM_EMB, 256)
        self.TNEWS_layer2 = nn.Linear(256, 15)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        # print(input_ids, ocnli_ids, ocemotion_ids, tnews_ids)
        if ocnli_ids.size()[0] > 0:
            # print(cls_emb.size())
            ocnli_out = self.denselayer(cls_emb[ocnli_ids, :])
            # ocnli_out = self.relu(ocnli_out)
            # ocnli_out = self.gelu(ocnli_out)
            # ocnli_out = self.dropout(ocnli_out)

            # ocnli_out = self.relu(self.OCNLI_layer1(ocnli_out))
            ocnli_out = self.OCNLI_layer2(ocnli_out)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            ocemotion_out = self.denselayer(cls_emb[ocemotion_ids, :])
            # ocemotion_out = self.relu(ocemotion_out)
            # ocemotion_out = self.gelu(ocemotion_out)
            # ocemotion_out = self.dropout(ocemotion_out)
            # ocemotion_out = self.relu(self.OCEMOTION_layer1(ocemotion_out))
            ocemotion_out = self.OCEMOTION_layer2(ocemotion_out)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            tnews_out = self.denselayer(cls_emb[tnews_ids, :])
            # tnews_out = self.relu(tnews_out)
            # tnews_out = self.gelu(tnews_out)
            # tnews_out = self.dropout(tnews_out)
            # tnews_out = self.relu(self.TNEWS_layer1(tnews_out))
            tnews_out = self.TNEWS_layer2(tnews_out)
        else:
            tnews_out = None
        # print(ocnli_out.size(),ocemotion_out.size(),tnews_out.size())
        return ocnli_out, ocemotion_out, tnews_out