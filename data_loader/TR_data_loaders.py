from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseDataLoader
import pickle
import numpy as np
import json
import os
from tqdm import tqdm
from torch._six import string_classes
import re
import random
import copy
import itertools
import math
from multiprocessing import Pool
from functools import partial

import pdb

from model.transformers import BertTokenizer

def tokenize_table(table, config):
    caption, headers, cells = table
    tokenized_meta = config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for _, z in headers]
    tokenized_cells = [[index, config.tokenizer.encode(cell, max_length=config.max_cell_length, add_special_tokens=False)] for index, cell in cells]

    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tok += list(itertools.chain(*tokenized_headers))
    input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
    input_tok_type += [1]*sum(tokenized_headers_length)
    input_ent_text = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}
    for index, cell in cells:
        if len(cell)!=0:
            tokenized_cell = config.tokenizer.encode(cell, max_length=config.max_cell_length, add_special_tokens=False)
        if len(tokenized_cell)==0:
            continue
        input_ent_text.append([index,tokenized_cell])
        input_ent_type.append(3 if index[1] == 0 else 4)
        e_i = len(input_ent_text)-1
        if index[1] not in column_en_map:
            column_en_map[index[1]] = [e_i]
        else:
            column_en_map[index[1]].append(e_i)
        if index[0] not in row_en_map:
            row_en_map[index[0]] = [e_i]
        else:
            row_en_map[index[0]].append(e_i)
    input_length = len(input_tok) + len(input_ent_text)
    meta_and_headers_length = tokenized_meta_length+sum(tokenized_headers_length)
    assert len(input_tok) == meta_and_headers_length
    #create input mask
    meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent_text)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent_text)], dtype=int)
    start_i = 0
    header_span = {}
    for h_i, (h_j, _) in enumerate(headers):
        header_span[h_j] = (start_i, start_i+tokenized_headers_length[h_i])
        start_i += tokenized_headers_length[h_i]
    for e_i, (index, _) in enumerate(input_ent_text):
        if index[1] in header_span:
            header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    ent_header_mask = np.transpose(header_ent_mask)

    input_tok_ent_mask = np.concatenate([meta_ent_mask, header_ent_mask], axis=0)
    ent_meta_mask = np.ones([len(input_ent_text), tokenized_meta_length], dtype=int)
    
    ent_ent_mask = np.eye(len(input_ent_text), dtype=int)
    for _,e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1

    for _,e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]

    
    input_ent_cell_length = [len(x) if len(x)!=0 else 1 for _,x in input_ent_text]
    if len(input_ent_cell_length) != 0:
        max_cell_length = max(input_ent_cell_length)
    else:
        max_cell_length = 0
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    if max_cell_length != 0:
        for i,(_,x) in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(x)] = x
    assert input_ent_mask[0].shape[1] == len(input_tok)
    return [input_tok,input_tok_type,input_tok_pos,input_tok_ent_mask,len(input_tok), \
                input_ent_text_padded,input_ent_cell_length,np.array(input_ent_type),input_ent_mask,len(input_ent_text)]

def tokenize_table_for_bert(table, config):
    caption, headers, cells = table
    tokenized_meta = config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for _, z in headers]

    input_tok = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tok += list(itertools.chain(*tokenized_headers))
    return [input_tok,len(input_tok)]

def process_single_TR(input_data, config):
    _, (q_id, tokenized_query, pos, neg) = input_data
    pos_inputs, neg_inputs = [], []
    for table in pos:
        input_tok,input_tok_type,input_tok_pos,input_tok_ent_mask,input_tok_length, \
                input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length = table
        prepend_input_tok = [config.tokenizer.cls_token_id]
        prepend_input_tok_pos = [0]
        prepend_input_tok_type = [5]
        tokenized_query_length = len(tokenized_query)
        prepend_input_tok += tokenized_query
        prepend_input_tok_pos += list(range(tokenized_query_length))
        prepend_input_tok_type += [5]*tokenized_query_length

        input_tok = prepend_input_tok+input_tok
        input_tok_pos = prepend_input_tok_pos + input_tok_pos
        input_tok_type = prepend_input_tok_type + input_tok_type
        input_tok_length += tokenized_query_length + 1
        input_tok_ent_mask = np.concatenate([np.ones([tokenized_query_length + 1, len(input_ent_text)], dtype=int), input_tok_ent_mask], axis=0)
        input_ent_mask = (np.concatenate([np.ones([len(input_ent_text), tokenized_query_length + 1], dtype=int), input_ent_mask[0]], axis=1), input_ent_mask[1])
        pos_inputs.append([np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),input_tok_ent_mask,input_tok_length, \
                input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length])
    for table in neg:
        input_tok,input_tok_type,input_tok_pos,input_tok_ent_mask,input_tok_length, \
                input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length = table
        prepend_input_tok = [config.tokenizer.cls_token_id]
        prepend_input_tok_pos = [0]
        prepend_input_tok_type = [5]
        tokenized_query_length = len(tokenized_query)
        prepend_input_tok += tokenized_query
        prepend_input_tok_pos += list(range(tokenized_query_length))
        prepend_input_tok_type += [5]*tokenized_query_length

        input_tok = prepend_input_tok+input_tok
        input_tok_pos = prepend_input_tok_pos + input_tok_pos
        input_tok_type = prepend_input_tok_type + input_tok_type
        input_tok_length += tokenized_query_length + 1
        input_tok_ent_mask = np.concatenate([np.ones([tokenized_query_length + 1, len(input_ent_text)], dtype=int), input_tok_ent_mask], axis=0)
        input_ent_mask = (np.concatenate([np.ones([len(input_ent_text), tokenized_query_length + 1], dtype=int), input_ent_mask[0]], axis=1), input_ent_mask[1])
        neg_inputs.append([np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),input_tok_ent_mask,input_tok_length, \
                input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length])

    return [q_id, pos_inputs, neg_inputs]

def process_single_TR_forBert(input_data, config):
    _, (q_id, tokenized_query, pos, neg) = input_data
    pos_inputs, neg_inputs = [], []
    for table in pos:
        input_tok,input_tok_length = table
        prepend_input_tok = [config.tokenizer.cls_token_id]
        tokenized_query_length = len(tokenized_query)
        prepend_input_tok += tokenized_query + [config.tokenizer.sep_token_id]
        prepend_input_tok_type = [0]*(tokenized_query_length+2)

        input_tok_type = prepend_input_tok_type + [1]*len(input_tok)
        input_tok = prepend_input_tok+input_tok
        
        input_tok_length += tokenized_query_length + 2
        pos_inputs.append([np.array(input_tok),np.array(input_tok_type),input_tok_length])
    for table in neg:
        input_tok,input_tok_length = table
        prepend_input_tok = [config.tokenizer.cls_token_id]
        tokenized_query_length = len(tokenized_query)
        prepend_input_tok += tokenized_query + [config.tokenizer.sep_token_id]
        prepend_input_tok_type = [0]*(tokenized_query_length+2)

        input_tok_type = prepend_input_tok_type + [1]*len(input_tok)
        input_tok = prepend_input_tok+input_tok
        
        input_tok_length += tokenized_query_length + 2
        neg_inputs.append([np.array(input_tok),np.array(input_tok_type),input_tok_length])

    return [q_id, pos_inputs, neg_inputs]

class WebQueryTableDataset(Dataset):

    def _preprocess(self, data_dir):
        if not self.for_bert:
            preprocessed_filename = os.path.join(
                data_dir, "procressed_TR", self.src
            )
        else:
            preprocessed_filename = os.path.join(
                data_dir, "procressed_TR_Bert", self.src
            )
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                if not self.for_bert:
                    os.mkdir(os.path.join(data_dir, "procressed_TR"))
                else:
                    os.mkdir(os.path.join(data_dir, "procressed_TR_Bert"))
            except FileExistsError:
                pass
            train_q_t, dev_q_t, test_q_t = {}, {}, {}
            with open(os.path.join(data_dir, "WQT.dataset.query.tsv"), "r", encoding='utf-8') as f:
                next(f)
                for i, q in enumerate(tqdm(f)):
                    q = q.strip().split('\t')
                    if q[2] == 'train':
                        train_q_t[i] = [i, self.tokenizer.encode(q[1], max_length=self.max_query_length, add_special_tokens=False), [], []]
                    elif q[2] == 'dev':
                        dev_q_t[i] = [i, self.tokenizer.encode(q[1], max_length=self.max_query_length, add_special_tokens=False), [], []]
                    elif q[2] == 'test':
                        test_q_t[i] = [i, self.tokenizer.encode(q[1], max_length=self.max_query_length, add_special_tokens=False), [], []]
                    else:
                        pdb.set_trace()
                    assert i == int(q[0])
            with open(os.path.join(data_dir, "WQT.dataset.table.tsv"), "r", encoding='utf-8') as f:
                next(f)
                tables = []
                for q, t in enumerate(tqdm(f)):
                    t = t.strip().split('\t')
                    if len(t) != 7:
                        pdb.set_trace()
                    cells = []
                    headers = []
                    for j,h in enumerate(t[4].split('_|_')):
                        if j >= 5:
                            break
                        h = h.strip()
                        if len(h)!=0:
                            headers.append([j,h])
                    column_num = j+1
                    rows = t[5].split('_||_')
                    for i,row in enumerate(rows):
                        if i >= 20:
                            break
                        for j, cell in enumerate(row.split('_|_')):
                            if j >= 5:
                                break
                            cell = cell.strip()
                            if cell != 'None' and len(cell)!=0:
                                cells.append([(i,j),cell])
                        if j+1 != column_num:
                            # pdb.set_trace()
                            cells = []
                            break
                    tables.append([
                        t[2] +' '+t[3],
                        headers,
                        cells
                    ])
                    assert q == int(t[0])
                pool = Pool(processes=5)
                tokenized_tables = list(tqdm(pool.imap(partial(tokenize_table if not self.for_bert else tokenize_table_for_bert,config=self), tables, chunksize=1000),total=len(tables)))
                pool.close()
            
            with open(os.path.join(data_dir, "WQT.dataset.query-table.tsv"), "r", encoding='utf-8') as f:
                next(f)
                pos = 0.0
                neg = 0.0
                for qt in tqdm(f):
                    qt = qt.strip().split('\t')
                    q_id = int(qt[0])
                    t_id = int(qt[1])
                    label = int(qt[2])
                    if label == 1:
                        pos += 1
                        if q_id in train_q_t:
                            train_q_t[q_id][2].append(tokenized_tables[t_id])
                        elif q_id in dev_q_t:
                            dev_q_t[q_id][2].append(tokenized_tables[t_id])
                        elif q_id in test_q_t:
                            test_q_t[q_id][2].append(tokenized_tables[t_id])
                        else:
                            pdb.set_trace()
                    else:
                        neg += 1
                        if q_id in train_q_t:
                            train_q_t[q_id][3].append(tokenized_tables[t_id])
                        elif q_id in dev_q_t:
                            dev_q_t[q_id][3].append(tokenized_tables[t_id])
                        elif q_id in test_q_t:
                            test_q_t[q_id][3].append(tokenized_tables[t_id])
                        else:
                            pdb.set_trace()
                    
        print('{} train pairs, {} dev pairs, {} test pairs'.format(len(train_q_t),len(dev_q_t),len(test_q_t)))
        print('pos/neg ratio: %f'%(pos/neg))
        pool = Pool(processes=4)
        processed_train_q_t = list(tqdm(pool.imap(partial(process_single_TR if not self.for_bert else process_single_TR_forBert,config=self), list(train_q_t.items()), chunksize=1000),total=len(train_q_t)))
        processed_dev_q_t = list(tqdm(pool.imap(partial(process_single_TR if not self.for_bert else process_single_TR_forBert,config=self), list(dev_q_t.items()), chunksize=1000),total=len(dev_q_t)))
        processed_test_q_t = list(tqdm(pool.imap(partial(process_single_TR if not self.for_bert else process_single_TR_forBert,config=self), list(test_q_t.items()), chunksize=1000),total=len(test_q_t)))
        pool.close()
        # pdb.set_trace()

        
        with open(os.path.join(data_dir, "procressed_TR" if not self.for_bert else "procressed_TR_Bert", 'train.pickle'), 'wb') as f:
            pickle.dump(processed_train_q_t, f)
        with open(os.path.join(data_dir, "procressed_TR" if not self.for_bert else "procressed_TR_Bert", 'dev.pickle'), 'wb') as f:
            pickle.dump(processed_dev_q_t, f)
        with open(os.path.join(data_dir, "procressed_TR" if not self.for_bert else "procressed_TR_Bert", 'test.pickle'), 'wb') as f:
            pickle.dump(processed_test_q_t, f)
        # pdb.set_trace()
        if self.src == "train":
            return processed_train_q_t
        elif self.src == "dev":
            return processed_dev_q_t
        else:
            return processed_test_q_t

    def __init__(self, data_dir, max_input_tok=500, src="train", max_length = [50, 50, 10, 10], force_new=False, tokenizer = None, for_bert=False):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
        self.for_bert = for_bert
        self.max_query_length = max_length[0]
        self.max_title_length = max_length[1]
        self.max_header_length = max_length[2]
        self.max_cell_length = max_length[3]
        self.force_new = force_new
        self.max_input_tok = max_input_tok
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class finetune_collate_fn_TR:
    def __init__(self, tokenizer, is_train=True, neg_num = 5):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.neg_num = neg_num
    def __call__(self, raw_batch):
        batch_q_id, batch_pos_inputs, batch_neg_inputs = zip(*raw_batch)
        if self.is_train:
            batch_neg_inputs = [random.sample(x, self.neg_num) if len(x)>self.neg_num else x for x in batch_neg_inputs]

        batch_size = len(batch_q_id)
        new_batch_q_id = []
        batch_input_tok,batch_input_tok_type,batch_input_tok_pos,batch_input_tok_ent_mask,batch_input_tok_length,batch_labels = [], [], [], [], [], []
        batch_input_ent_text,batch_input_ent_cell_length,batch_input_ent_type,batch_input_ent_mask,batch_input_ent_length = [], [], [], [], []
        for i in range(batch_size):
            if not self.is_train and (len(batch_pos_inputs[i]) == 0 or len(batch_neg_inputs[i]) == 0):
                continue
            if len(batch_pos_inputs[i]) != 0:
                input_tok,input_tok_type,input_tok_pos,input_tok_ent_mask,input_tok_length, \
                    input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length = zip(*batch_pos_inputs[i])
                batch_input_tok += list(input_tok)
                batch_input_tok_type += list(input_tok_type)
                batch_input_tok_pos += list(input_tok_pos)
                batch_input_tok_length += list(input_tok_length)
                batch_input_tok_ent_mask += list(input_tok_ent_mask)
                batch_input_ent_text += list(input_ent_text)
                batch_input_ent_cell_length += list(input_ent_cell_length)
                batch_input_ent_type += list(input_ent_type)
                batch_input_ent_mask += list(input_ent_mask)
                batch_input_ent_length += list(input_ent_length)
                batch_labels += [1]*len(input_tok)
                new_batch_q_id += [batch_q_id[i]]*len(input_tok)
            if len(batch_neg_inputs[i]) != 0:
                input_tok,input_tok_type,input_tok_pos,input_tok_ent_mask,input_tok_length, \
                    input_ent_text,input_ent_cell_length,input_ent_type,input_ent_mask,input_ent_length = zip(*batch_neg_inputs[i])
                batch_input_tok += list(input_tok)
                batch_input_tok_type += list(input_tok_type)
                batch_input_tok_pos += list(input_tok_pos)
                batch_input_tok_length += list(input_tok_length)
                batch_input_tok_ent_mask += list(input_tok_ent_mask)
                batch_input_ent_text += list(input_ent_text)
                batch_input_ent_cell_length += list(input_ent_cell_length)
                batch_input_ent_type += list(input_ent_type)
                batch_input_ent_mask += list(input_ent_mask)
                batch_input_ent_length += list(input_ent_length)
                batch_labels += [0]*len(input_tok)
                new_batch_q_id += [batch_q_id[i]]*len(input_tok)
        batch_q_id = new_batch_q_id
        batch_size = len(batch_q_id)
        max_input_tok_length = max(batch_input_tok_length)
        max_input_ent_length = max(batch_input_ent_length)
        max_input_ent_cell_length = max([max(z) if len(z)!=0 else 0 for z in batch_input_ent_cell_length])
        

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)

        batch_input_ent_text_padded = np.zeros([batch_size, max_input_ent_length, max_input_ent_cell_length], dtype=int)
        batch_input_ent_text_length = np.ones([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)

        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length, max_input_tok_length+max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros([batch_size, max_input_ent_length, max_input_tok_length+max_input_ent_length], dtype=int)
        for i, (tok_l, ent_l) in enumerate(zip(batch_input_tok_length, batch_input_ent_length)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_tok_mask_padded[i, :tok_l, :tok_l] = 1
            batch_input_tok_mask_padded[i, :tok_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_tok_ent_mask[i]
            batch_input_ent_text_padded[i, :ent_l, :batch_input_ent_text[i].shape[-1]] = batch_input_ent_text[i]
            batch_input_ent_text_length[i, :ent_l] = batch_input_ent_cell_length[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_mask_padded[i, :ent_l, :tok_l] = batch_input_ent_mask[i][0]
            batch_input_ent_mask_padded[i, :ent_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_ent_mask[i][1]
        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_input_ent_text_padded = torch.LongTensor(batch_input_ent_text_padded)
        batch_input_ent_text_length = torch.LongTensor(batch_input_ent_text_length)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)
        batch_labels = torch.FloatTensor(batch_labels)

        return batch_q_id, batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded,\
            batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, batch_labels

class finetune_collate_fn_TR_forBert:
    def __init__(self, tokenizer, is_train=True, neg_num = 5):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.neg_num = neg_num
    def __call__(self, raw_batch):
        batch_q_id, batch_pos_inputs, batch_neg_inputs = zip(*raw_batch)
        if self.is_train:
            batch_neg_inputs = [random.sample(x, self.neg_num) if len(x)>self.neg_num else x for x in batch_neg_inputs]

        batch_size = len(batch_q_id)
        new_batch_q_id = []
        batch_input_tok,batch_input_tok_type,batch_input_tok_pos,batch_input_tok_ent_mask,batch_input_tok_length,batch_labels = [], [], [], [], [], []
        batch_input_ent_text,batch_input_ent_cell_length,batch_input_ent_type,batch_input_ent_mask,batch_input_ent_length = [], [], [], [], []
        for i in range(batch_size):
            if not self.is_train and (len(batch_pos_inputs[i]) == 0 or len(batch_neg_inputs[i]) == 0):
                continue
            if len(batch_pos_inputs[i]) != 0:
                input_tok,input_tok_type,input_tok_length = zip(*batch_pos_inputs[i])
                batch_input_tok += list(input_tok)
                batch_input_tok_type += list(input_tok_type)
                batch_input_tok_length += list(input_tok_length)
                batch_labels += [1]*len(input_tok)
                new_batch_q_id += [batch_q_id[i]]*len(input_tok)
            if len(batch_neg_inputs[i]) != 0:
                input_tok,input_tok_type,input_tok_length = zip(*batch_neg_inputs[i])
                batch_input_tok += list(input_tok)
                batch_input_tok_type += list(input_tok_type)
                batch_input_tok_length += list(input_tok_length)
                batch_labels += [0]*len(input_tok)
                new_batch_q_id += [batch_q_id[i]]*len(input_tok)
        batch_q_id = new_batch_q_id
        batch_size = len(batch_q_id)
        max_input_tok_length = max(batch_input_tok_length)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)

        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        for i, tok_l in enumerate(batch_input_tok_length):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_mask_padded[i, :tok_l] = 1
        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor([])
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_input_ent_text_padded = torch.LongTensor([])
        batch_input_ent_text_length = torch.LongTensor([])
        batch_input_ent_type_padded = torch.LongTensor([])
        batch_input_ent_mask_padded = torch.LongTensor([])
        batch_labels = torch.FloatTensor(batch_labels)

        return batch_q_id, batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded,\
            batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, batch_labels

class TRLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        is_train = True,
        num_workers=0,
        sampler=None,
        neg_num=5,
        for_bert=False
    ):
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.is_train = is_train
        if not for_bert:
            self.collate_fn = finetune_collate_fn_TR(dataset.tokenizer, is_train=self.is_train, neg_num=neg_num)
        else:
            self.collate_fn = finetune_collate_fn_TR_forBert(dataset.tokenizer, is_train=self.is_train, neg_num=neg_num)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)
