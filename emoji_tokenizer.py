from transformers import BertModel, BertTokenizer
import json

import re
from collections import defaultdict


def additional_emoji():
    additional_emojis = []
    emojis = defaultdict(int)
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']: # 
        with open('./tianchi_datasets/' + e + '/total.csv') as f:
            for line in f:
                emoji = re.compile('\[.{1,5}?\]')
                sentence = line.strip().split()[1]
                tokens = emoji.findall(sentence)
                # print(tokens)
                if len(tokens)>0:
                    for token in set(tokens):
                        emojis[token] += 1
    for k in emojis:
        if emojis[k]>20:
            additional_emojis.append(k)
    return additional_emojis

def load_tokenizer_emoji(path_or_name):
    tokenizer = BertTokenizer.from_pretrained(path_or_name)

    ocemotion_test = dict()
    with open('./tianchi_datasets/OCEMOTION/test.json') as f:
        for line in f:
            ocemotion_test = json.loads(line)
            break
    ocnli_test = dict()
    with open('./tianchi_datasets/OCNLI/test.json') as f:
        for line in f:
            ocnli_test = json.loads(line)
            break

    emojis = additional_emoji()
    print(f'add {len(emojis)} emoji to tokenizer.')
    special_tokens_dict = {'additional_special_tokens': emojis}
    tokenizer.add_special_tokens(special_tokens_dict)
    # print(len(tokenizer))
    return tokenizer

if __name__ =="__main__":
    sentence = '[伤心]'
    tokenizer = load_tokenizer('./roberta_pretrain_model')
    flower1 = tokenizer([sentence], add_special_tokens=True, padding=True, return_tensors='pt')
    print(flower1)