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
from torch._six import container_abcs, string_classes, int_classes, FileNotFoundError
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
    pgTitle,secTitle,caption,headers,rows = input_table
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
                    rows
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
                        tmp_rows
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
    pgEnt,pgTitle,secTitle,caption,headers,core_entities,entities = input_table
    tokenized_meta = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)+\
                    config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
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
    meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
    start_i = 0
    header_span = {}
    for h_i, (h_j, _) in enumerate(headers):
        header_span[h_j] = (start_i, start_i+tokenized_headers_length[h_i])
        start_i += tokenized_headers_length[h_i]
    for e_i, (index, _) in enumerate(entities):
        header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
    ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)
    ent_header_mask = np.transpose(header_ent_mask)
    ent_ent_mask = np.zeros([len(input_ent), len(input_ent)], dtype=int)
    for _,e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    for _,e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]

    # prepend pgEnt to input_ent
    if pgEnt!=-1:
        input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    else:
        input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    input_ent = [pgEnt if pgEnt!=-1 else 0] + input_ent
    input_ent_type = [2] + input_ent_type
    input_ent_pos = [0] + input_ent_pos
    new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int), np.ones([len(input_ent), len(input_ent)], dtype=int)]
    new_input_ent_mask[0][1:, :] = input_ent_mask[0]
    new_input_ent_mask[1][1:, 1:] = input_ent_mask[1]
    if pgEnt==-1:
        new_input_ent_mask[1][:, 0] = 0
    input_ent_mask = new_input_ent_mask
    core_entity_mask = [0]+core_entity_mask

    core_entity_set = list(set([z[1] for z in core_entities]))
    all_entity_set = list(set([z for z in input_ent]))
    def find_id(e, e_list):
        for i, e_1 in enumerate(e_list):
            if e == e_1:
                return i
        raise Exception
    input_ent_local_id = [find_id(e, all_entity_set) for e in input_ent]

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
    return [np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),(np.array(input_tok_mask[0]),np.array(input_tok_mask[1])),len(input_tok), \
                np.array(input_ent),np.array(input_ent_local_id),np.array(input_ent_type),np.array(input_ent_pos),(np.array(input_ent_mask[0]),np.array(input_ent_mask[1])),len(input_ent), \
                np.array(core_entity_mask),core_entity_set,all_entity_set]
class WikiEntityTableDataset(Dataset):

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
            origin_data = open(os.path.join(data_dir, self.src + "_tables.jsonl"), "r")
        print("Pre-processing data...")
        origin_table_num = 0
        actual_tables = []
        table_removed = 0
        for table in tqdm(origin_data):
            origin_table_num += 1
            table = json.loads(table.strip())
            pgTitle = table.get("pgTitle", "").lower()
            pgEnt = table.get("pgId", [-1])
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
            num_rows = len(rows)
            entities = []
            split = [0]
            tmp_entity_num = 0
            for i in range(num_rows):
                for j in entity_columns:
                    if entity_cells[i,j] == 1:
                        try:
                            entity = self.entity_wikid2id[rows[i][j]['surfaceLinks'][0]['target']['id']]
                        except:
                            continue
                        entities.append([(i,j), entity])
                        tmp_entity_num += 1
                        if j == 0:
                            core_entities.append([(i,j), entity])
                if tmp_entity_num >= self.max_cell:
                    split.append(len(entities))
                    tmp_entity_num = 0
            if split[-1]!=len(entities):
                split.append(len(entities))
            if len(core_entities) == 0:
                table_removed += 1
                continue
            for i in range(len(split)-1):
                actual_tables.append([
                    pgEnt,
                    pgTitle,
                    secTitle,
                    caption,
                    headers,
                    core_entities,
                    entities[split[i]:split[i+1]]
                ])
            
        actual_table_num = len(actual_tables)
        print('%d original tables, actual %d tables in total\n%d tables removed because of extra entity filtering'%(origin_table_num, actual_table_num, table_removed))
        
        pool = Pool(processes=4) 
        processed_data = list(tqdm(pool.imap(partial(process_single_entity_table,config=self), actual_tables, chunksize=1000),total=len(actual_tables)))
        pool.close()
        # pdb.set_trace()

        
        with open(preprocessed_filename, 'wb') as f:
            pickle.dump(processed_data, f)
        # pdb.set_trace()
        return processed_data

    def __init__(self, data_dir, entity_vocab, max_cell=100, max_input_tok=350, max_input_ent=150, src="train", max_length = [50, 10, 10], force_new=False, tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
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

def mask_ent(inputs_origin, inputs_local_id, core_entity_mask, entity_wikid2id, mlm_probability=0.15):
    """ Prepare masked entities inputs/labels for masked entity modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs_local_id.clone()
    inputs = inputs_origin.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [list(map(lambda x: 1 if x == entity_wikid2id['[PAD]'] else 0, val)) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input ent with [ENT_MASK]/[PG_ENT_MASK]/[CORE_ENT_MASK] accordingly
    pg_ent_mask = torch.zeros(labels.shape)
    pg_ent_mask[:,0] = 1
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = entity_wikid2id['[ENT_MASK]']
    inputs[indices_replaced & pg_ent_mask.bool()] = entity_wikid2id['[PG_ENT_MASK]']
    inputs[indices_replaced & core_entity_mask.bool()] = entity_wikid2id['[CORE_ENT_MASK]']

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(low=RESERVED_ENT_VOCAB_NUM,high=len(entity_wikid2id), size=labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
class pretrain_entity_table_collate_fn:
    def __init__(self, tokenizer, entity_wikid2id, mlm_probability, ent_mlm_probability, max_entity_candidate=1000):
        self.tokenizer = tokenizer
        self.entity_wikid2id = entity_wikid2id
        self.mlm_probability = mlm_probability
        self.ent_mlm_probability = ent_mlm_probability
        self.max_entity_candidate = max_entity_candidate

    def generate_random_candidate(self, batch_size, indice_mask):
        random_shifts = np.random.random((batch_size, len(self.entity_wikid2id)))
        random_shifts[indice_mask] = 10
        return np.argpartition(random_shifts, self.max_entity_candidate, axis=1)[:, :self.max_entity_candidate]
        
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
        batch_input_tok,batch_input_tok_type,batch_input_tok_pos,batch_input_tok_mask,batch_input_tok_length, \
            batch_input_ent,batch_input_ent_local_id,batch_input_ent_type,batch_input_ent_pos,batch_input_ent_mask,batch_input_ent_length, \
            batch_core_entity_mask,batch_core_entity_set,batch_all_entity_set = zip(*raw_batch)

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
        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_input_ent_padded = torch.LongTensor(batch_input_ent_padded)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_pos_padded = torch.LongTensor(batch_input_ent_pos_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)
        batch_core_entity_mask_padded = torch.LongTensor(batch_core_entity_mask_padded)
        batch_input_ent_local_id_padded = torch.LongTensor(batch_input_ent_local_id_padded)
        
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
        batch_input_ent_final, batch_input_ent_labels = mask_ent(batch_input_ent_padded, batch_input_ent_local_id_padded, batch_core_entity_mask_padded, self.entity_wikid2id, mlm_probability=self.ent_mlm_probability)

        #random sample candidate
        indice_mask = (list(itertools.chain(*[[i]*len(x) for i,x in enumerate(batch_all_entity_set)])), list(itertools.chain(*batch_all_entity_set)))
        batch_candidate_entity_set = self.generate_random_candidate(batch_size, indice_mask)
        for i in range(batch_size):
            batch_candidate_entity_set[i,:len(batch_all_entity_set[i])] = batch_all_entity_set[i]
        batch_candidate_entity_set = torch.LongTensor(batch_candidate_entity_set)

        # pdb.set_trace()

        return batch_input_tok_final, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_labels, batch_input_tok_mask_padded, \
                batch_input_ent_final, batch_input_ent_type_padded, batch_input_ent_pos_padded, batch_input_ent_labels, batch_input_ent_mask_padded, batch_candidate_entity_set

class EntityTableLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self,
        dataset,
        batch_size,
        max_entity_candidate=1000,
        shuffle=True,
        num_workers=0,
        mlm_probability=0.15,
        ent_mlm_probability=0.15,
        sampler=None
    ):
        self.mlm_probability = mlm_probability
        self.ent_mlm_probability = ent_mlm_probability
        self.max_entity_candidate=max_entity_candidate
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.collate_fn = pretrain_entity_table_collate_fn(dataset.tokenizer, dataset.entity_wikid2id, self.mlm_probability, self.ent_mlm_probability, self.max_entity_candidate)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)

