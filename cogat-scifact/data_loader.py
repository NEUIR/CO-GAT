import random, os
import argparse
import re

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import NLLLoss
import logging
import json
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import jsonlines

class data_loader(Dataset):
    def __init__(self, args, data_path,tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.args = args
        self.project_dim = args.project_dim
        self.sep_tokid = tokenizer.sep_token_id
        self.pad_tokid = tokenizer.pad_token_id
        self.cls_tokid = tokenizer.cls_token_id
        self.label_map = {'SUPPORT': 0, 'CONTRADICT': 1, 'NOT_ENOUGH_INFO': 2}
        self.read_file()

    def process_sent(self, sentence):
        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub("LRB", " ( ", title)
        title = re.sub("RRB", " )", title)
        title = re.sub("COLON", ":", title)
        return title

    def read_file(self):
        self.data = []
        with open(self.data_path) as fin:
            for line in fin:
                example = json.loads(line.strip())
                self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, data):
        batch_ids = []
        batch_dids = []
        batch_label = []
        inp_padding_inputs = []
        msk_padding_inputs = []
        seg_padding_inputs = []
        batch_ground_truth = []
        process_batch = {}
        for step, example in enumerate(data):
            ground_truth_list = []
            evi_list = []
            ids = example['id']
            batch_ids.append(ids)

            label = example['label']

            label = self.label_map[label]
            batch_label.append(label)
            claim = example['claim']
            evi_list.append((self.process_sent(claim)))
            if len(example['evidence']) != 0:
                for evidence in example['evidence']:
                    dids = evidence[0]
                    batch_dids.append(dids)
                    # evi_list.append((self.process_sent(claim), self.process_sent(evidence[1] + evidence[3])))
                    evi_list.append((self.process_sent(claim), self.process_sent(evidence[3])))
                    evi_list = evi_list[:self.evi_num]
                    if evidence[4] == 1:
                        ground_truth = 1
                    else:
                        ground_truth = 0
                    ground_truth_list.append(ground_truth)
                if len(example['evidence']) < self.evi_num - 1:
                    for i in range(self.evi_num - 1 - len(example['evidence'])):
                        ground_truth = 0
                        ground_truth_list.append(ground_truth)
            else:
                for i in range(self.evi_num - 1):
                    evi_list.append((self.process_sent(claim)))
                    ground_truth = 0
                    ground_truth_list.append(ground_truth)
            batch_ground_truth.append(ground_truth_list)
            batch_data = self.tokenizer.batch_encode_plus(evi_list, max_length=self.max_len, padding='longest',
                                                          truncation='only_second')
            input_ids = [input_id + [self.pad_tokid] * (self.max_len - len(input_id)) for input_id in batch_data["input_ids"]]
            input_mask = [mask + [0] * (self.max_len - len(mask)) for mask in batch_data["attention_mask"]]
            input_seg = [seq + [0] * (self.max_len - len(seq)) for seq in batch_data["token_type_ids"]]
            # 证据不足五条也补0
            inp_padding = input_ids[:self.evi_num]
            msk_padding = input_mask[:self.evi_num]
            seg_padding = input_seg[:self.evi_num]
            inp_padding += ([[0] * self.max_len] * (self.evi_num - len(inp_padding)))
            msk_padding += ([[0] * self.max_len] * (self.evi_num - len(msk_padding)))
            seg_padding += ([[0] * self.max_len] * (self.evi_num - len(seg_padding)))
            inp_padding_inputs += inp_padding
            msk_padding_inputs += msk_padding
            seg_padding_inputs += seg_padding


        process_batch["input_ids"] = torch.LongTensor(inp_padding_inputs)
        process_batch["attention_mask"] = torch.LongTensor(msk_padding_inputs)
        process_batch["token_type_ids"] = torch.LongTensor(seg_padding_inputs)
        process_batch["labels"] = torch.LongTensor(batch_label)
        process_batch["ground_truth"] = torch.LongTensor(batch_ground_truth)
        process_batch["ids"] = batch_ids
        process_batch["dids"] = batch_dids
        return process_batch

class data_loader_test(Dataset):
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.args = args
        self.project_dim = args.project_dim
        self.sep_tokid = tokenizer.sep_token_id
        self.pad_tokid = tokenizer.pad_token_id
        self.cls_tokid = tokenizer.cls_token_id
        self.label_map = {'SUPPORT': 0, 'CONTRADICT': 1, 'NOT_ENOUGH_INFO': 2}
        self.read_file()

    def process_sent(self, sentence):
        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub("LRB", " ( ", title)
        title = re.sub("RRB", " )", title)
        title = re.sub("COLON", ":", title)
        return title

    def read_file(self):
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(self.args.corpus)}
        evidence_retrieval = jsonlines.open(self.args.evidence_retrieval)
        dataset = jsonlines.open(self.args.dataset)
        self.data = []
        for data, retrieval in list(zip(dataset, evidence_retrieval)):
            assert data['id'] == retrieval['claim_id']
            for did, sids in retrieval["evidence"].items():
                evidence_list = list()
                content = corpus[int(did)]
                evidence_list.append((self.process_sent(data["claim"])))
                for sid in sids:
                    evidence = content["abstract"][sid]
                    title = content["title"]
                    evidence_list.append((self.process_sent(data["claim"]), self.process_sent(evidence)))
                if len(sids) < self.evi_num - 1:
                    for i in range(self.evi_num - 1 - len(sids)):
                        evidence_list.append((self.process_sent(data["claim"])))
                self.data.append([int(data['id']), did, sids, evidence_list])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, data):
        batch_ids = []
        batch_dids = []
        inp_padding_inputs = []
        msk_padding_inputs = []
        seg_padding_inputs = []
        process_batch = {}
        for step, example in enumerate(data):
            evi_list = []
            ids = example[0]
            batch_ids.append(ids)
            dids = example[1]
            batch_dids.append(dids)
            evi_list = example[3]
            batch_data = self.tokenizer.batch_encode_plus(evi_list, max_length=self.max_len, padding='longest', truncation='only_second')
            input_ids = [input_id + [self.pad_tokid] * (self.max_len - len(input_id)) for input_id in
                         batch_data["input_ids"]]
            input_mask = [mask + [0] * (self.max_len - len(mask)) for mask in batch_data["attention_mask"]]
            input_seg = [seq + [0] * (self.max_len - len(seq)) for seq in batch_data["token_type_ids"]]
            # 证据不足五条也补0
            inp_padding = input_ids[:self.evi_num]
            msk_padding = input_mask[:self.evi_num]
            seg_padding = input_seg[:self.evi_num]
            inp_padding += ([[0] * self.max_len] * (self.evi_num - len(inp_padding)))
            msk_padding += ([[0] * self.max_len] * (self.evi_num - len(msk_padding)))
            seg_padding += ([[0] * self.max_len] * (self.evi_num - len(seg_padding)))
            inp_padding_inputs += inp_padding
            msk_padding_inputs += msk_padding
            seg_padding_inputs += seg_padding


        process_batch["input_ids"] = torch.LongTensor(inp_padding_inputs)
        process_batch["attention_mask"] = torch.LongTensor(msk_padding_inputs)
        process_batch["token_type_ids"] = torch.LongTensor(seg_padding_inputs)
        process_batch["ids"] = batch_ids
        process_batch["dids"] = batch_dids
        return process_batch
