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

def process_single_table(input_table, config):
    pgTitle,secTitle,caption,headers,rows,core_cand = input_table
    tokenized_meta = config.tokenizer.encode(pgTitle, max_length=config.max_title_length)+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length)+\
                    config.tokenizer.encode(caption, max_length=config.max_title_length)
    tokenized_headers = [config.tokenizer.encode(z,max_length=config.max_header_length) for z in headers]
    row_num = len(rows)
    column_num = len(rows[0])
    input_tok = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tok += list(itertools.chain(*tokenized_headers))
    tokenized_rows = []
    for row in rows:
        tokenized_rows.append([config.tokenizer.encode(str(z),max_length=config.max_cell_length) for z in row])
        input_tok += list(itertools.chain(*tokenized_rows[-1]))
    tokenized_rows_length = [[len(z) for z in tokenized_row] for tokenized_row in tokenized_rows]
    input_length = len(input_tok)
    assert input_length<=config.max_tokenized_input, "max set as %d, run into input with length %d"%(config.max_tokenized_input,input_length)
    meta_and_headers_length = tokenized_meta_length+sum(tokenized_headers_length)
    input_type = np.array([0]*meta_and_headers_length+[1]*(input_length-meta_and_headers_length))
    #create input mask
    #create base mask blocks
    header_cell_mask = []
    for i in range(row_num):
        tmp_mask = np.zeros([sum(tokenized_headers_length), sum(tokenized_rows_length[i])], dtype=int)
        last_i = 0
        last_j = 0
        for j in range(column_num):
            tmp_mask[last_i:last_i+tokenized_rows_length[i][j],last_j:last_j+tokenized_headers_length[j]] = 1
            last_i += tokenized_rows_length[i][j]
            last_j += tokenized_headers_length[j]
        header_cell_mask.append(tmp_mask)
    cell_cell_mask = []
    for i_0 in range(row_num):
        cell_cell_mask.append([])
        for i_1 in range(row_num):
            if i_0 == i_1:
                cell_cell_mask[i_0].append(np.ones([sum(tokenized_rows_length[i_0]), sum(tokenized_rows_length[i_1])],dtype=int))
            elif i_1 > i_0:
                tmp_mask = np.zeros([sum(tokenized_rows_length[i_0]), sum(tokenized_rows_length[i_1])],dtype=int)
                last_i = 0
                last_j = 0
                for j in range(column_num):
                    tmp_mask[last_i:last_i+tokenized_rows_length[i_0][j],last_j:last_j+tokenized_rows_length[i_1][j]] = 1
                    last_i += tokenized_rows_length[i_0][j]
                    last_j += tokenized_rows_length[i_1][j]
                cell_cell_mask[i_0].append(tmp_mask)
            else:
                cell_cell_mask[i_0].append(np.transpose(cell_cell_mask[i_1][i_0]))
    #meta can attend to all tokens
    meta_mask = np.ones([tokenized_meta_length, input_length],dtype=int)
    #headers can attend to meta/all headers and content in same column
    header_mask = np.concatenate([
        np.ones([sum(tokenized_headers_length),meta_and_headers_length],dtype=int),
        np.concatenate(header_cell_mask, axis=1)
    ], axis=1)
    #each cell can attend to cells in the same row or column
    cell_mask = []
    for i in range(row_num):
        row_mask = np.concatenate([
            np.ones([sum(tokenized_rows_length[i]),tokenized_meta_length], dtype=int),
            np.transpose(header_cell_mask[i]),
            np.concatenate(cell_cell_mask[i], axis=1)
        ],axis=1)
        cell_mask.append(row_mask)
    cell_mask = np.concatenate(cell_mask, axis=0)
    input_mask = np.concatenate([meta_mask,header_mask,cell_mask],axis=0)
    return [input_tok,input_type,input_mask]
class WikiTableDataset(Dataset):
    def _preprocess(self, data_dir):
        preprocessed_filename = os.path.join(
            data_dir, "procressed", self.src
        )
        preprocessed_filename += ".pickle"
        print("try loading preprocessed data from %s" % preprocessed_filename)
        if not self.force_new and os.path.exists(preprocessed_filename):
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            try:
                os.mkdir(os.path.join(data_dir, "procressed"))
            except FileExistsError:
                pass
            origin_data = open(os.path.join(data_dir, self.src + ".tables.jsonl"), "r")
        print("Pre-processing data...")
        origin_table_num = 0
        actual_tables = []
        for table in tqdm(origin_data):
            origin_table_num += 1
            table = json.loads(table.strip())
            pgTitle = table.get("page_title", "").lower()
            secTitle = table.get("section_title", "").lower()
            caption = table.get("caption", "").lower()
            headers = table.get("header", [])
            rows = table.get("rows", {})
            num_rows = len(rows)
            num_columns = len(rows[0])
            if num_rows*num_columns <= self.max_cell:
                actual_tables.append([
                    pgTitle,
                    secTitle,
                    caption,
                    headers,
                    rows,
                ])
            else:
                row_limit = max([1,int(num_rows/math.ceil(num_rows/(self.max_cell/num_columns)))])
                rest_rows = rows
                while len(rest_rows) != 0:
                    tmp_rows = rest_rows[:row_limit]
                    rest_rows = rest_rows[row_limit:]
                    actual_tables.append([
                        pgTitle,
                        secTitle,
                        caption,
                        headers,
                        tmp_rows,
                    ])
        actual_table_num = len(actual_tables)
        print('%d original tables, actual %d tables in total'%(origin_table_num, actual_table_num))
        
        pool = Pool(processes=4) 
        processed_data = list(tqdm(pool.imap(partial(process_single_table,config=self), actual_tables, chunksize=1000),total=len(actual_tables)))
        pool.close()
        # pdb.set_trace()

        
        with open(preprocessed_filename, 'wb') as f:
            pickle.dump(processed_data, f)
        # pdb.set_trace()
        return processed_data

    def __init__(self, data_dir, max_cell=30.0, max_tokenized_input=512, src="train", max_length = [50, 10, 10], force_new=False, tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.src = src
        self.max_cell = float(max_cell)
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.max_cell_length = max_length[2]
        self.force_new = force_new
        self.max_tokenized_input = max_tokenized_input
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [list(map(lambda x: 1 if x == tokenizer.pad_token_id else 0, val)) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class pretrain_table_collate_fn:
    def __init__(self, tokenizer, mlm_probability):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
    def __call__(self, raw_batch):
        batch_input_tok, batch_input_type, batch_input_mask = zip(*raw_batch)
        batch_input_tok = [torch.LongTensor(x) for x in batch_input_tok]
        batch_input_type = [torch.LongTensor(x) for x in batch_input_type]
        batch_input_mask = [torch.FloatTensor(x) for x in batch_input_mask]
        #pad sequence
        batch_size = len(batch_input_tok)
        batch_max_length = max([len(x) for x in batch_input_tok])
        batch_input_tok_padded = torch.nn.utils.rnn.pad_sequence(batch_input_tok, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_input_type_padded = torch.nn.utils.rnn.pad_sequence(batch_input_type, batch_first=True)
        batch_input_mask_padded = batch_input_mask[0].data.new(batch_size,batch_max_length,batch_max_length).fill_(0)
        for i in range(batch_size):
            current_length = len(batch_input_tok[i])
            batch_input_mask_padded[i,:current_length,:current_length] = batch_input_mask[i]
        batch_input_tok_final, labels = mask_tokens(batch_input_tok_padded, self.tokenizer, mlm_probability=self.mlm_probability)
            
        return batch_input_tok_final, batch_input_type_padded, batch_input_mask_padded, labels

class TableLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        num_workers=0,
        mlm_probability=0.15,
        sampler=None
    ):
        self.mlm_probability = mlm_probability
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.collate_fn = pretrain_table_collate_fn(dataset.tokenizer, self.mlm_probability)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)

RESERVED_ENT_VOCAB = {0:{'wiki_id':'[PAD]'},
                        1:{'wiki_id':'[ENT_MASK]'},
                        2:{'wiki_id':'[PG_ENT_MASK]'},
                        3:{'wiki_id':'[CORE_ENT_MASK]'}
                        }
RESERVED_ENT_VOCAB_NUM = len(RESERVED_ENT_VOCAB)

def process_single_entity_table(input_table, config):
    table_id,pgEnt,pgTitle,secTitle,caption,headers,core_entities,all_entities,entities,entity_cand = input_table
    tokenized_meta = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for _,z in headers]
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
    input_ent = []
    input_ent_pos = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}
    core_entity_mask = []
    for e_i, (index, entity) in enumerate(entities):
        input_ent.append(entity)
        input_ent_type.append(3 if index[1] == 0 else 4)
        input_ent_pos.append(0)
        core_entity_mask.append(1 if index[1]==0 else 0)
        if index[1] not in column_en_map:
            column_en_map[index[1]] = [e_i]
        else:
            column_en_map[index[1]].append(e_i)
        if index[0] not in row_en_map:
            row_en_map[index[0]] = [e_i]
        else:
            row_en_map[index[0]].append(e_i)
    input_length = len(input_tok) + len(input_ent)
    assert len(input_tok) < config.max_input_tok
    assert len(input_ent) < config.max_input_ent
    meta_and_headers_length = tokenized_meta_length+sum(tokenized_headers_length)
    assert len(input_tok) == meta_and_headers_length
    #create input mask
    tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
    if config.src == "train":
        meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
    else:
        meta_ent_mask = np.zeros([tokenized_meta_length, len(input_ent)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
    start_i = 0
    header_span = {}
    for h_i, (h_j, _) in enumerate(headers):
        header_span[h_j] = (start_i, start_i+tokenized_headers_length[h_i])
        start_i += tokenized_headers_length[h_i]
    for e_i, (index, _) in enumerate(entities):
        header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    ent_header_mask = np.transpose(header_ent_mask)
    if config.src != "train":
        header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)

    input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
    ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)
    
    ent_ent_mask = np.eye(len(input_ent), dtype=int)
    for _,e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                if config.src=="train" or (e_i_2<e_i_1 and input_ent[e_i_2]>=len(RESERVED_ENT_VOCAB)):
                    ent_ent_mask[e_i_1, e_i_2] = 1

    for _,e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                if config.src=="train" or (e_i_2<e_i_1 and input_ent[e_i_2]>=len(RESERVED_ENT_VOCAB)):
                    ent_ent_mask[e_i_1, e_i_2] = 1
    input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]

    # prepend [CORE_ENT_MASK] to input, input_ent[1] = [CORE_ENT_MASK]
    input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    input_ent = [config.entity_wikid2id['[CORE_ENT_MASK]']] + input_ent
    input_ent_type = [3] + input_ent_type
    input_ent_pos = [0] + input_ent_pos
    # prepend pgEnt to input_ent, input_ent[0] = pgEnt
    if pgEnt!=-1:
        input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    else:
        input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    input_ent = [pgEnt if pgEnt!=-1 else 0] + input_ent
    input_ent_type = [2] + input_ent_type
    input_ent_pos = [0] + input_ent_pos

    new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int), np.ones([len(input_ent), len(input_ent)], dtype=int)]
    new_input_ent_mask[0][2:, :] = input_ent_mask[0]
    new_input_ent_mask[1][2:, 2:] = input_ent_mask[1]
    # process [CORE_ENT_MASK] mask
    new_input_ent_mask[0][1, tokenized_meta_length:] = 0
    if 0 in header_span:
        assert header_span[0][0] == 0
        new_input_ent_mask[0][1, tokenized_meta_length+header_span[0][0]:tokenized_meta_length+header_span[0][1]] = 1
    new_input_ent_mask[1][1, 2:] = 0
    new_input_ent_mask[1][2:, 1] = 0
    if 0 in column_en_map:
        new_input_ent_mask[1][1, 2+column_en_map[0][0]] = 1 # seed=1
    # process pgEnt mask
    if pgEnt==-1:
        new_input_ent_mask[1][:, 0] = 0
        new_input_ent_mask[1][0, :] = 0
    input_ent_mask = new_input_ent_mask
    core_entity_mask = [0,1]+core_entity_mask
    
    if entity_cand is not None:
        entity_cand = list(set(entity_cand)-all_entities)
    all_entity_set = list(all_entities)

    def find_id(e, e_list):
        for i, e_1 in enumerate(e_list):
            if e == e_1:
                return i
        # pdb.set_trace()
        raise Exception
    if config.src == "train":
        input_ent_local_id = [find_id(pgEnt, all_entity_set) if pgEnt!=-1 else 0,0]+[find_id(e, all_entity_set) for e in input_ent[2:]]
        exclusive_ent_mask = np.full([len(input_ent), max([len(z) for _,z in column_en_map.items()])-1], 1000) # mask entity in the same row for prediction
        for e_i, (index, _) in enumerate(entities):
            tmp_j = 0
            for e_i_0 in column_en_map[index[1]]:
                if input_ent_local_id[2+e_i_0] != input_ent_local_id[2+e_i]:
                    exclusive_ent_mask[2+e_i,tmp_j] = input_ent_local_id[2+e_i_0]
                    tmp_j += 1

    else:
        input_ent_local_id = [find_id(pgEnt, all_entity_set) if pgEnt!=-1 else 0,0]
        i = 2
        while i < len(input_ent):
            if input_ent[i]>=len(RESERVED_ENT_VOCAB):
                e = input_ent[i]
                input_ent_local_id.append(find_id(e, all_entity_set))
                i+=1
            else:
                e = input_ent[i+1]
                tmp_e = find_id(e, all_entity_set)
                input_ent_local_id += [tmp_e, tmp_e]
                i+=2
        exclusive_ent_mask = None

    if len(core_entities) > 1:
        core_ent_local_id = [find_id(e, all_entity_set) for e in core_entities[1:]]
    else:
        core_ent_local_id = []

    # input_tok_padded = input_tok + [config.tokenizer.pad_token_id]*(config.max_input_tok-len(input_tok))
    # input_tok_type_padded = input_tok_type + [0]*(config.max_input_tok-len(input_tok_type))
    # input_tok_pos_padded = input_tok_pos + [0]*(config.max_input_tok-len(input_tok_pos))
    # input_tok_mask_padded = [np.zeros([config.max_input_tok, config.max_input_tok], dtype=int), np.zeros([config.max_input_tok, config.max_input_ent], dtype=int)]
    # input_tok_mask_padded[0][:input_tok_mask[0].shape[0],:input_tok_mask[0].shape[1]] = input_tok_mask[0]
    # input_tok_mask_padded[1][:input_tok_mask[1].shape[0],:input_tok_mask[1].shape[1]] = input_tok_mask[1]

    # input_ent_padded = input_ent + [0]*(config.max_input_ent-len(input_ent))
    # input_ent_local_id_padded = input_ent_local_id + [0]*(config.max_input_ent-len(input_ent_local_id))
    # input_ent_type_padded = input_ent_type + [0]*(config.max_input_ent-len(input_ent_type))
    # input_ent_pos_padded = input_ent_pos + [0]*(config.max_input_ent-len(input_ent_pos))
    # input_ent_mask_padded = [np.zeros([config.max_input_ent, config.max_input_tok], dtype=int), np.zeros([config.max_input_ent, config.max_input_ent], dtype=int)]
    # input_ent_mask_padded[0][:input_ent_mask[0].shape[0],:input_ent_mask[0].shape[1]] = input_ent_mask[0]
    # input_ent_mask_padded[1][:input_ent_mask[1].shape[0],:input_ent_mask[1].shape[1]] = input_ent_mask[1]
    # core_entity_mask_padded = core_entity_mask + [0]*(config.max_input_ent-len(core_entity_mask))
    # return [input_tok_padded,input_tok_type_padded,input_tok_pos_padded,input_tok_mask_padded,len(input_tok), \
    #             input_ent_padded,input_ent_local_id_padded,input_ent_type_padded,input_ent_pos_padded,input_ent_mask_padded,len(input_ent), \
    #             core_entity_mask_padded,core_entity_set,all_entity_set]
    return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),(np.array(input_tok_mask[0]),np.array(input_tok_mask[1])),len(input_tok), \
                np.array(input_ent),np.array(input_ent_local_id),np.array(input_ent_type),np.array(input_ent_pos),(np.array(input_ent_mask[0]),np.array(input_ent_mask[1])),len(input_ent), \
                np.array(core_entity_mask),core_ent_local_id,all_entity_set,entity_cand,exclusive_ent_mask]

def process_single_entity_table_CER(input_table, config):
    table_id,pgEnt,pgTitle,secTitle,caption,headers,core_entities,all_entities,entities,entity_cand = input_table
    tokenized_meta = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_header = config.tokenizer.encode(headers[0][1], max_length=config.max_header_length, add_special_tokens=False)
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    tokenized_header_length = len(tokenized_header)
    input_tok += tokenized_header
    input_tok_pos += list(range(tokenized_header_length))
    input_tok_type += [1]*tokenized_header_length
    input_ent = [entity for _, entity in entities]
    assert len(input_tok) < config.max_input_tok
    assert len(input_ent) < config.max_input_ent
    meta_and_header_length = tokenized_meta_length+tokenized_header_length
    assert len(input_tok) == meta_and_header_length
    def find_id(e, e_list):
        for i, e_1 in enumerate(e_list):
            if e == e_1:
                return i
        # pdb.set_trace()
        raise Exception
    # prepend special token to input_ent, input_ent[0] = pgEnt, input_ent[1] = [CORE_ENT_MASK]
    input_ent_local_id = [find_id(e, core_entities) for e in input_ent]
    input_ent = [pgEnt if pgEnt!=-1 else 0, config.entity_wikid2id['[CORE_ENT_MASK]']] + input_ent

    return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),len(input_tok), \
                np.array(input_ent),len(input_ent), np.array(input_ent_local_id), np.array(core_entities), \
                entity_cand]

class WikiEntityTableDataset(Dataset):

    def _preprocess(self, data_dir):
        if self.mode == 0:
            preprocessed_filename = os.path.join(
                data_dir, "procressed", self.src
            )
        elif self.mode == 1:
            preprocessed_filename = os.path.join(
                data_dir, "procressed_CER", self.src
            )
        else:
            raise Exception
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                if self.mode == 0:
                    os.mkdir(os.path.join(data_dir, "procressed"))
                else:
                    os.mkdir(os.path.join(data_dir, "procressed_CER"))
            except FileExistsError:
                pass
            origin_data = open(os.path.join(data_dir, self.src + "_tables.jsonl"), "r")
            entity_candidate_file = os.path.join(data_dir, self.src + ".entity_candidate.pkl")
            if os.path.exists(entity_candidate_file):
                with open(entity_candidate_file, "rb") as f:
                    entity_candidate = pickle.load(f)
            else:
                entity_candidate = None
        print("Pre-processing data...")
        origin_table_num = 0
        actual_tables = []
        table_removed = 0
        for table in tqdm(origin_data):
            origin_table_num += 1
            table = json.loads(table.strip())
            table_id = table.get("_id","")
            pgTitle = table.get("pgTitle", "").lower()
            pgEnt = table.get("pgId", -1)
            if entity_candidate is not None:
                entity_cand = entity_candidate.get(table_id, [])
                entity_cand = [self.entity_wikid2id[z] for z in entity_cand if z in self.entity_wikid2id]
            else:
                entity_cand = None

            if pgEnt != -1:
                try:
                    pgEnt = self.entity_wikid2id[pgEnt]
                except:
                    pgEnt = -1
            secTitle = table.get("sectionTitle", "").lower()
            caption = table.get("tableCaption", "").lower()
            headers = table.get("processed_tableHeaders", [])
            rows = table.get("tableData", {})
            entity_columns = table.get("entityColumn", [])
            headers = [[j, headers[j]] for j in entity_columns]
            entity_cells = np.array(table.get("entityCell",[[]]))
            core_entities = []
            all_entities = set()
            if pgEnt!=-1:
                all_entities.add(pgEnt)
            num_rows = len(rows)
            num_columns = len(rows[0])
            entities = []
            split = [0]
            tmp_entity_num = 0
            for i in range(num_rows):
                tmp_entities = []
                for j in range(num_columns):
                    if j in entity_columns:
                        if self.mode == 1 and j!=0:
                            continue
                        if entity_cells[i,j] == 1:
                            try:
                                entity = self.entity_wikid2id[rows[i][j]['surfaceLinks'][0]['target']['id']]
                                entity_cells[i,j] = entity
                                tmp_entities.append([(i,j), entity])
                            except:
                                entity_cells[i,j] = 0
                    else:
                        entity_cells[i,j] = 0
                if len(tmp_entities) == 0:
                    continue
                if self.mode == 0:
                    if i == 0 or not (entity_cells[i] == entity_cells[:i]).all(axis=1).any():
                        has_core = True if tmp_entities[0][0][1]==0 else False
                        if has_core or self.src == "train":
                            for index, entity in tmp_entities:
                                if self.mode == 0 and self.src != "train" and index[1]!=0:
                                    entities.append([index, self.entity_wikid2id['[ENT_MASK]']])
                                    tmp_entity_num += 1
                                entities.append([index, entity])
                                all_entities.add(entity)
                                tmp_entity_num += 1
                                if index[1] == 0:
                                    core_entities.append(entity)
                            if tmp_entity_num >= self.max_cell:
                                split.append(len(entities))
                                tmp_entity_num = 0
                elif self.mode == 1:
                    for index, entity in tmp_entities:
                        entities.append([index, entity])
                        tmp_entity_num += 1
                        core_entities.append(entity)
                    if tmp_entity_num >= self.max_cell:
                        split.append(len(entities))
                        tmp_entity_num = 0
                #     pdb.set_trace()
            if split[-1]!=len(entities):
                split.append(len(entities))
            if len(core_entities) < 5:
                if self.src!="train" or len(core_entities) == 0:# or (self.mode == 1 and len(core_entities) < 3):
                    table_removed += 1
                    continue
            if split[-2]!=0 and split[-1]-split[-2]<5:
                split[-2] = split[-1]-5
            for i in range(len(split)-1):
                actual_tables.append([
                    table_id,
                    pgEnt,
                    pgTitle,
                    secTitle,
                    caption,
                    headers,
                    core_entities,
                    all_entities,
                    entities[split[i]:split[i+1]],
                    entity_cand
                ])
            
        actual_table_num = len(actual_tables)
        print('%d original tables, actual %d tables in total\n%d tables removed because of extra entity filtering'%(origin_table_num, actual_table_num, table_removed))
        
        pool = Pool(processes=4)
        if self.mode == 0:
            processed_data = list(tqdm(pool.imap(partial(process_single_entity_table,config=self), actual_tables, chunksize=1000),total=len(actual_tables)))
        elif self.mode == 1:
            processed_data = list(tqdm(pool.imap(partial(process_single_entity_table_CER,config=self), actual_tables, chunksize=1000),total=len(actual_tables)))
        # elif self.mode == 2:
        else:
            raise Exception
        pool.close()
        # pdb.set_trace()

        
        with open(preprocessed_filename, 'wb') as f:
            pickle.dump(processed_data, f)
        # pdb.set_trace()
        return processed_data

    def __init__(self, data_dir, entity_vocab, max_cell=100, max_input_tok=350, max_input_ent=150, src="train", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
        self.mode = mode #{0:pretrain,1:core entity retrieval,2:cell filling}
        self.max_cell = float(max_cell)
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.max_cell_length = max_length[2]
        self.force_new = force_new
        self.max_input_tok = max_input_tok
        self.max_input_ent = max_input_ent
        self.entity_vocab = entity_vocab
        self.entity_wikid2id = {self.entity_vocab[x]['wiki_id']:x for x in self.entity_vocab}
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def mask_ent(inputs_origin, inputs_local_id, core_entity_mask, entity_wikid2id, mlm_probability=0.15, is_train=False):
    """ Prepare masked entities inputs/labels for masked entity modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs_local_id.clone()
    inputs = inputs_origin.clone()
    if is_train:
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = (inputs<len(RESERVED_ENT_VOCAB))#[list(map(lambda x: 1 if x == entity_wikid2id['[PAD]'] else 0, val)) for val in labels.tolist()]
        # special_tokens_mask[:, 1] = True
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens
    
        # 80% of the time, we replace masked input ent with [ENT_MASK]/[PG_ENT_MASK]/[CORE_ENT_MASK] accordingly
        pg_ent_mask = torch.zeros(labels.shape)
        pg_ent_mask[:,0] = 1
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = entity_wikid2id['[ENT_MASK]']
        inputs[indices_replaced & pg_ent_mask.bool()] = entity_wikid2id['[PG_ENT_MASK]']
        inputs[indices_replaced & core_entity_mask] = entity_wikid2id['[CORE_ENT_MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=RESERVED_ENT_VOCAB_NUM,high=len(entity_wikid2id), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        inputs[:, 1] = inputs_origin[:, 1]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    else:
        labels[inputs!=entity_wikid2id['[ENT_MASK]']] = -1
    return inputs, labels
class pretrain_entity_table_collate_fn:
    def __init__(self, tokenizer, entity_wikid2id, mlm_probability, ent_mlm_probability, max_entity_candidate=1000, is_train=True, candidate_distribution=None, use_cand=True):
        self.tokenizer = tokenizer
        self.entity_wikid2id = entity_wikid2id
        self.mlm_probability = mlm_probability
        self.ent_mlm_probability = ent_mlm_probability
        self.max_entity_candidate = max_entity_candidate
        self.is_train = is_train
        self.candidate_distribution = candidate_distribution
        self.use_cand = use_cand

    def generate_random_candidate(self, batch_size, indice_mask):
        random_shifts = np.random.random((batch_size, len(self.entity_wikid2id)))
        random_shifts[indice_mask] = 10
        return np.argpartition(random_shifts, self.max_entity_candidate, axis=1)[:, :self.max_entity_candidate]
    
    def generate_random_candidate_v2(self, batch_size, masked_entity, candidate_distribution=None, ent_cand=None):
        random_shifts = np.random.random(len(self.entity_wikid2id))
        if candidate_distribution is not None:
            random_shifts /= np.sum(random_shifts)
            random_shifts -= candidate_distribution
        all_masked = list(itertools.chain(*masked_entity))
        random_shifts[all_masked] = 10
        all_masked = set(all_masked)
        final_candidates = np.tile(np.argpartition(random_shifts, self.max_entity_candidate)[:self.max_entity_candidate],[batch_size, 1])
        for i, masked in enumerate(masked_entity):
            final_candidates[i, :len(masked)] = masked
            if self.use_cand:
                cand_i = ent_cand[i]
                if len(cand_i)+len(masked) > self.max_entity_candidate:
                    cand_i = random.sample(cand_i, self.max_entity_candidate-len(masked))
                final_candidates[i, len(masked):len(masked)+len(cand_i)] = cand_i
            else:
                remain = list(all_masked-set(masked))
                final_candidates[i, len(masked):len(masked)+len(remain)] = remain
        return final_candidates
        
    def __call__(self, raw_batch):
        # batch_input_tok_padded,batch_input_tok_type_padded,batch_input_tok_pos_padded,batch_input_tok_mask_padded,input_tok_length, \
        #     batch_input_ent_padded,batch_input_ent_local_id_padded,batch_input_ent_type_padded,batch_input_ent_pos_padded,batch_input_ent_mask_padded,input_ent_length, \
        #     batch_core_entity_mask_padded,batch_core_entity_set,batch_all_entity_set = zip(*raw_batch)

        # max_input_tok_length = max(input_tok_length)
        # max_input_ent_length = max(input_ent_length)

        # batch_input_tok_padded = torch.LongTensor([x[:max_input_tok_length] for x in batch_input_tok_padded])
        # batch_input_tok_type_padded = torch.LongTensor([x[:max_input_tok_length] for x in batch_input_tok_type_padded])
        # batch_input_tok_pos_padded = torch.LongTensor([x[:max_input_tok_length] for x in batch_input_tok_pos_padded])
        # batch_input_tok_tok_mask_padded, batch_input_tok_ent_mask_padded = zip(*batch_input_tok_mask_padded)
        # batch_input_tok_tok_mask_padded = torch.FloatTensor([x[:max_input_tok_length, :max_input_tok_length] for x in batch_input_tok_tok_mask_padded])
        # batch_input_tok_ent_mask_padded = torch.FloatTensor([x[:max_input_tok_length, :max_input_ent_length] for x in batch_input_tok_ent_mask_padded])
        

        # # the first is pgEnt
        # batch_input_ent_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_padded])
        # batch_input_ent_type_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_type_padded])
        # batch_input_ent_pos_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_pos_padded])
        # batch_input_ent_tok_mask_padded, batch_input_ent_ent_mask_padded = zip(*batch_input_ent_mask_padded)
        # batch_input_ent_tok_mask_padded = torch.FloatTensor([x[:max_input_ent_length, :max_input_tok_length] for x in batch_input_ent_tok_mask_padded])
        # batch_input_ent_ent_mask_padded = torch.FloatTensor([x[:max_input_ent_length, :max_input_ent_length] for x in batch_input_ent_ent_mask_padded])
        # batch_core_entity_mask_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_core_entity_mask_padded])
        # batch_input_ent_local_id_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_local_id_padded])
        

        
        # batch_size = len(batch_input_tok_padded)
        # batch_input_tok_mask_padded = torch.cat([batch_input_tok_tok_mask_padded, batch_input_tok_ent_mask_padded], dim=2)
        # batch_input_ent_mask_padded = torch.cat([batch_input_ent_tok_mask_padded, batch_input_ent_ent_mask_padded], dim=2)
        batch_table_id,batch_input_tok,batch_input_tok_type,batch_input_tok_pos,batch_input_tok_mask,batch_input_tok_length, \
            batch_input_ent,batch_input_ent_local_id,batch_input_ent_type,batch_input_ent_pos,batch_input_ent_mask,batch_input_ent_length, \
            batch_core_entity_mask,batch_core_ent_local_id,batch_all_entity_set,batch_entity_cand,batch_exclusive_ent_mask = zip(*raw_batch)
        
        if batch_entity_cand[0] is None and self.use_cand:
            raise Exception

        max_input_tok_length = max(batch_input_tok_length)
        max_input_ent_length = max(batch_input_ent_length)
        batch_size = len(batch_input_tok_length)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_input_ent_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_pos_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros([batch_size, max_input_ent_length, max_input_tok_length+max_input_ent_length], dtype=int)
        batch_core_entity_mask_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_local_id_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        if self.is_train:
            max_input_col_ent_num = max([z.shape[1] for z in batch_exclusive_ent_mask])
            batch_exclusive_ent_mask_padded = np.full([batch_size, max_input_ent_length, max_input_col_ent_num], 1000, dtype=int)

        for i, (tok_l, ent_l) in enumerate(zip(batch_input_tok_length, batch_input_ent_length)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_tok_mask_padded[i, :tok_l, :tok_l] = batch_input_tok_mask[i][0]
            batch_input_tok_mask_padded[i, :tok_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_tok_mask[i][1]

            batch_input_ent_padded[i, :ent_l] = batch_input_ent[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_pos_padded[i, :ent_l] = batch_input_ent_pos[i]
            batch_input_ent_mask_padded[i, :ent_l, :tok_l] = batch_input_ent_mask[i][0]
            batch_input_ent_mask_padded[i, :ent_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_ent_mask[i][1]
            batch_core_entity_mask_padded[i, :ent_l] = batch_core_entity_mask[i]
            batch_input_ent_local_id_padded[i, :ent_l] = batch_input_ent_local_id[i]
            if self.is_train:
                batch_exclusive_ent_mask_padded[i, :ent_l, :batch_exclusive_ent_mask[i].shape[1]] = batch_exclusive_ent_mask[i]
        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_input_ent_padded = torch.LongTensor(batch_input_ent_padded)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_pos_padded = torch.LongTensor(batch_input_ent_pos_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)
        batch_core_entity_mask_padded = torch.BoolTensor(batch_core_entity_mask_padded)
        batch_input_ent_local_id_padded = torch.LongTensor(batch_input_ent_local_id_padded)
        if self.is_train:
            batch_exclusive_ent_mask_padded = torch.LongTensor(batch_exclusive_ent_mask_padded)
        else:
            batch_exclusive_ent_mask_padded = None
        
        # # the first is pgEnt
        # batch_input_ent_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_padded])
        # batch_input_ent_type_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_type_padded])
        # batch_input_ent_pos_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_pos_padded])
        # batch_input_ent_tok_mask_padded, batch_input_ent_ent_mask_padded = zip(*batch_input_ent_mask_padded)
        # batch_input_ent_tok_mask_padded = torch.FloatTensor([x[:max_input_ent_length, :max_input_tok_length] for x in batch_input_ent_tok_mask_padded])
        # batch_input_ent_ent_mask_padded = torch.FloatTensor([x[:max_input_ent_length, :max_input_ent_length] for x in batch_input_ent_ent_mask_padded])
        # batch_core_entity_mask_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_core_entity_mask_padded])
        # batch_input_ent_local_id_padded = torch.LongTensor([x[:max_input_ent_length] for x in batch_input_ent_local_id_padded])


        batch_input_tok_final, batch_input_tok_labels = mask_tokens(batch_input_tok_padded, self.tokenizer, mlm_probability=self.mlm_probability)
        batch_input_ent_final, batch_input_ent_labels = mask_ent(batch_input_ent_padded, batch_input_ent_local_id_padded, batch_core_entity_mask_padded, self.entity_wikid2id, mlm_probability=self.ent_mlm_probability, is_train=self.is_train)

        #random sample candidate
        # indice_mask = (list(itertools.chain(*[[i]*len(x) for i,x in enumerate(batch_all_entity_set)])), list(itertools.chain(*batch_all_entity_set)))
        # batch_candidate_entity_set = self.generate_random_candidate(batch_size, indice_mask)
        batch_candidate_entity_set = self.generate_random_candidate_v2(batch_size, batch_all_entity_set, self.candidate_distribution, ent_cand=batch_entity_cand)
        batch_candidate_entity_set = torch.LongTensor(batch_candidate_entity_set)
        if not self.is_train:
            batch_core_entity_label = np.zeros([batch_size, self.max_entity_candidate], dtype=bool)
            for i in range(batch_size):
                batch_core_entity_label[i, batch_core_ent_local_id[i]] = True
            batch_core_entity_label = torch.BoolTensor(batch_core_entity_label)
        else:
            batch_core_entity_label = None
        return batch_table_id,batch_input_tok_final, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_labels, batch_input_tok_mask_padded, \
                batch_input_ent_final, batch_input_ent_type_padded, batch_input_ent_pos_padded, batch_input_ent_labels, batch_input_ent_mask_padded, batch_candidate_entity_set, batch_core_entity_label, batch_exclusive_ent_mask_padded, batch_core_entity_mask_padded

class pretrain_entity_table_collate_fn_CER(pretrain_entity_table_collate_fn):
    def __init__(self, tokenizer, entity_wikid2id, mlm_probability, ent_mlm_probability, max_entity_candidate=1000, is_train=True, candidate_distribution=None, use_cand=True, seed_num=1):
        self.tokenizer = tokenizer
        self.entity_wikid2id = entity_wikid2id
        self.mlm_probability = mlm_probability
        self.ent_mlm_probability = ent_mlm_probability
        self.max_entity_candidate = max_entity_candidate
        self.is_train = is_train
        self.candidate_distribution = candidate_distribution
        self.use_cand = use_cand
        self.seed = seed_num
    def __call__(self, raw_batch):
        batch_table_id,batch_input_tok,batch_input_tok_type,batch_input_tok_pos,batch_input_tok_length, \
            batch_input_ent,batch_input_ent_length, batch_input_ent_local_id, batch_core_entities,\
           batch_entity_cand = zip(*raw_batch)
        
        if batch_entity_cand[0] is None and self.use_cand:
            raise Exception

        max_input_tok_length = max(batch_input_tok_length)
        max_input_ent_length = max(batch_input_ent_length)
        batch_size = len(batch_input_tok_length)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)

        batch_input_ent_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded[:, 0] = 2
        batch_input_ent_pos_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_ent_mask_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_ent_mask_padded[:, 1] = 1

        batch_input_mask_padded = np.zeros([batch_size, 1, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_seed_ent = []
        batch_target_ent = np.full([batch_size, self.max_entity_candidate], 0, dtype=int)
        for i, (tok_l, ent_l) in enumerate(zip(batch_input_tok_length, batch_input_ent_length)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_ent_padded[i, :ent_l] = batch_input_ent[i]
            batch_input_ent_type_padded[i, 1:ent_l] = 3
            batch_input_ent_pos_padded[i, :ent_l] = 0

            if self.seed !=-1:
                tmp_cand_core = set(range(ent_l-2))
                tmp_selected_core = random.sample(tmp_cand_core,self.seed)
                batch_seed_ent.append(batch_input_ent_local_id[i][tmp_selected_core])
                tmp_cand_core = list(tmp_cand_core-set(tmp_selected_core))
                # batch_target_ent[i,:len(tmp_cand_core)] = batch_input_ent_local_id[i][tmp_cand_core]
                batch_target_ent[i,batch_input_ent_local_id[i][tmp_cand_core]] = 1
                batch_input_ent_ent_mask_padded[i,2:][tmp_selected_core] = 1
            else:
                batch_input_ent_ent_mask_padded[i,2:ent_l] = 1
            batch_input_ent_ent_mask_padded[i,0] = batch_input_ent[i][0]!=0

            batch_input_mask_padded[i, :, :tok_l] = 1

        batch_input_mask_padded[:,0,max_input_tok_length:] = batch_input_ent_ent_mask_padded        

        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)

        batch_input_ent_padded = torch.LongTensor(batch_input_ent_padded)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_pos_padded = torch.LongTensor(batch_input_ent_pos_padded)

        batch_input_mask_padded = torch.LongTensor(batch_input_mask_padded)

        batch_seed_ent = torch.LongTensor(batch_seed_ent)

        batch_target_ent = torch.LongTensor(batch_target_ent)

        #random sample candidate
        batch_candidate_entity_set = self.generate_random_candidate_v2(batch_size, batch_core_entities, self.candidate_distribution, ent_cand=batch_entity_cand)
        batch_candidate_entity_set = torch.LongTensor(batch_candidate_entity_set)
        return batch_table_id,batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, \
                batch_input_ent_padded, batch_input_ent_type_padded, batch_input_ent_pos_padded, \
                batch_input_mask_padded, batch_candidate_entity_set, batch_seed_ent, batch_target_ent

def CER_build_input(pgEnt, pgTitle, secTitle, caption, header, seed_entities, entity_cand, config):
    tokenized_meta = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_header = config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False)
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    tokenized_header_length = len(tokenized_header)
    input_tok += tokenized_header
    input_tok_pos += list(range(tokenized_header_length))
    input_tok_type += [1]*tokenized_header_length
    input_ent = [config.entity_wikid2id[entity] for entity in seed_entities]
    input_ent = [config.entity_wikid2id[pgEnt] if pgEnt!=-1 else 0, config.entity_wikid2id['[CORE_ENT_MASK]'], config.entity_wikid2id['[CORE_ENT_MASK]']] + input_ent

    input_mask = np.ones([1, len(input_tok)+len(input_ent), len(input_tok)+len(input_ent)], dtype=int)
    if input_ent[0]==0:
        input_mask[0,:,len(input_tok)] = 0
    input_mask[0,len(input_tok)+2,len(input_tok)+3:] = 0
    input_mask = torch.LongTensor(input_mask)
    input_tok_mask = input_mask[:,:len(input_tok),:]
    input_ent_mask = input_mask[:,len(input_tok):,:]

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])
    
    input_ent = torch.LongTensor([input_ent])
    input_ent_type = torch.full_like(input_ent, 3)
    input_ent_type[:, 0] = 2

    candidate_entity_set = [config.entity_wikid2id[entity] for entity in entity_cand]
    candidate_entity_set = torch.LongTensor([candidate_entity_set])

    return input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
            input_ent, input_ent_type, \
            input_ent_mask, candidate_entity_set


class EntityTableLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self,
        dataset,
        batch_size,
        max_entity_candidate=1000,
        sample_distribution=None,
        shuffle=True,
        is_train = True,
        num_workers=0,
        mlm_probability=0.15,
        ent_mlm_probability=0.15,
        sampler=None,
        use_cand=True,
        mode=0
    ):
        self.mlm_probability = mlm_probability
        self.ent_mlm_probability = ent_mlm_probability
        self.max_entity_candidate=max_entity_candidate
        self.shuffle = shuffle
        self.use_cand = use_cand
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.is_train = is_train
        self.sample_distribution = sample_distribution
        self.mode = mode
        if self.mode == 0:
            self.collate_fn = pretrain_entity_table_collate_fn(dataset.tokenizer, dataset.entity_wikid2id, self.mlm_probability, self.ent_mlm_probability, self.max_entity_candidate, is_train=self.is_train, candidate_distribution=self.sample_distribution, use_cand=self.use_cand)
        elif self.mode == 1:
            self.collate_fn = pretrain_entity_table_collate_fn_CER(dataset.tokenizer, dataset.entity_wikid2id, self.mlm_probability, self.ent_mlm_probability, self.max_entity_candidate, is_train=self.is_train, candidate_distribution=self.sample_distribution, use_cand=self.use_cand)
        else:
            raise Exception

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)
