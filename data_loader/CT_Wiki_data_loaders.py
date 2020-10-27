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


def process_single_CT(input_data, config):
    table_id, pgTitle, pgEnt, secTitle, caption, headers, entities, type_annotations = input_data
    entities = [z for column in entities for z in column[:config.max_column]]
    pgEnt = config.entity_wikid2id.get(pgEnt, -1)

    tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_meta = tokenized_pgTitle+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for z in headers]
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
    input_ent_text = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}
    for e_i, (index, cell) in enumerate(entities):
        entity, entity_text = cell
        entity = config.entity_wikid2id.get(entity, 0)
        tokenized_ent_text = config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False)
        input_ent.append(entity)
        input_ent_text.append(tokenized_ent_text)
        input_ent_type.append(4)
        if index[1] not in column_en_map:
            column_en_map[index[1]] = [e_i]
        else:
            column_en_map[index[1]].append(e_i)
        if index[0] not in row_en_map:
            row_en_map[index[0]] = [e_i]
        else:
            row_en_map[index[0]].append(e_i)
    input_ent_length = len(input_ent)
    # create column entity mask
    column_entity_mask = np.zeros([len(type_annotations), len(input_ent)], dtype=int)
    for j in range(len(type_annotations)):
        for e_i_1 in column_en_map[j]:
            column_entity_mask[j, e_i_1] = 1
    # create column header mask
    start_i = 0
    header_span = {}
    column_header_mask = np.zeros([len(type_annotations), len(input_tok)], dtype=int)
    for j in range(len(type_annotations)):
        header_span[j] = (start_i, start_i+tokenized_headers_length[j])
        column_header_mask[j, tokenized_meta_length+header_span[j][0]:tokenized_meta_length+header_span[j][1]] = 1
        start_i += tokenized_headers_length[j]
    #create input mask
    tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
    meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
    
    for e_i, (index, _) in enumerate(entities):
        header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    ent_header_mask = np.transpose(header_ent_mask)

    input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
    ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)
    
    ent_ent_mask = np.eye(len(input_ent), dtype=int)
    for _,e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    for _,e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]
    # prepend pgEnt to input_ent, input_ent[0] = pgEnt
    if pgEnt!=-1:
        input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    else:
        input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    input_ent = [pgEnt if pgEnt!=-1 else 0] + input_ent
    input_ent_text = [tokenized_pgTitle[:config.max_cell_length]] + input_ent_text
    input_ent_type = [2] + input_ent_type

    new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int), np.ones([len(input_ent), len(input_ent)], dtype=int)]
    new_input_ent_mask[0][1:, :] = input_ent_mask[0]
    new_input_ent_mask[1][1:, 1:] = input_ent_mask[1]
    if pgEnt==-1:
        new_input_ent_mask[1][:, 0] = 0
        new_input_ent_mask[1][0, :] = 0
    column_entity_mask = np.concatenate([np.zeros([len(type_annotations), 1], dtype=int),column_entity_mask],axis=1)

    input_ent_mask = new_input_ent_mask
    labels = np.zeros([len(type_annotations), config.type_num], dtype=int)
    for j, types in enumerate(type_annotations):
        for t in types:
            labels[j, config.type_vocab[t]] = 1
    input_ent_cell_length = [len(x) if len(x)!=0 else 1 for x in input_ent_text]
    max_cell_length = max(input_ent_cell_length)
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    for i,x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x

    return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),(np.array(input_tok_mask[0]),np.array(input_tok_mask[1])),len(input_tok), \
                np.array(input_ent),input_ent_text_padded,input_ent_cell_length,np.array(input_ent_type),(np.array(input_ent_mask[0]),np.array(input_ent_mask[1])),len(input_ent), \
                column_header_mask,column_entity_mask,labels,len(labels)]

class WikiCTDataset(Dataset):

    def _preprocess(self, data_dir):
        preprocessed_filename = os.path.join(
            data_dir, "procressed_WikiCT", self.src
        )
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                os.mkdir(os.path.join(data_dir, "procressed_WikiCT"))
            except FileExistsError:
                pass
            with open(os.path.join(data_dir, "{}.table_col_type.json".format(self.src)), "r") as f:
                cols = json.load(f)

        print('{} {} tables'.format(len(cols),self.src))
        pool = Pool(processes=4)
        processed_cols = list(tqdm(pool.imap(partial(process_single_CT,config=self), cols, chunksize=1000),total=len(cols)))
        pool.close()
        # pdb.set_trace()

        
        with open(os.path.join(data_dir, "procressed_WikiCT", '{}.pickle'.format(self.src)), 'wb') as f:
            pickle.dump(processed_cols, f)
        # pdb.set_trace()
        return processed_cols

    def __init__(self, data_dir, entity_vocab, type_vocab, max_column=10, max_input_tok=500, src="train", max_length = [50, 10, 10], force_new=False, tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
        self.force_new = force_new
        self.max_input_tok = max_input_tok
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.max_cell_length = max_length[2]
        self.max_column = max_column
        self.entity_vocab = entity_vocab
        self.entity_wikid2id = {self.entity_vocab[x]['wiki_id']:x for x in self.entity_vocab}
        self.type_vocab = type_vocab
        self.type_num = len(self.type_vocab)
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class finetune_collate_fn_CT:
    def __init__(self, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
    def __call__(self, raw_batch):
        batch_table_id, batch_input_tok, batch_input_tok_type, batch_input_tok_pos, batch_input_tok_mask, batch_input_tok_length, \
            batch_input_ent, batch_input_ent_text, batch_input_ent_cell_length, batch_input_ent_type, batch_input_ent_mask, batch_input_ent_length, \
            batch_column_header_mask, batch_column_entity_mask, batch_labels, batch_col_num = zip(*raw_batch)
        
        batch_size = len(batch_table_id)
        max_input_tok_length = max(batch_input_tok_length)
        max_input_ent_length = max(batch_input_ent_length)
        max_input_cell_length = max([z.shape[-1] for z in batch_input_ent_text])
        max_input_col_num = max(batch_col_num)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_input_ent_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_text_padded = np.zeros([batch_size, max_input_ent_length, max_input_cell_length], dtype=int)
        batch_input_ent_text_length = np.ones([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros([batch_size, max_input_ent_length, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_column_entity_mask_padded = np.zeros([batch_size, max_input_col_num, max_input_ent_length], dtype=int)
        batch_column_header_mask_padded = np.zeros([batch_size, max_input_col_num, max_input_tok_length], dtype=int)
        batch_labels_padded = np.zeros([batch_size, max_input_col_num, batch_labels[0].shape[-1]], dtype=int)
        batch_labels_mask = np.zeros([batch_size, max_input_col_num], dtype=int)
        
        for i, (tok_l, ent_l, col_num) in enumerate(zip(batch_input_tok_length, batch_input_ent_length, batch_col_num)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_tok_mask_padded[i, :tok_l, :tok_l] = batch_input_tok_mask[i][0]
            batch_input_tok_mask_padded[i, :tok_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_tok_mask[i][1]

            batch_input_ent_padded[i, :ent_l] = batch_input_ent[i]
            batch_input_ent_text_padded[i, :ent_l, :batch_input_ent_text[i].shape[-1]] = batch_input_ent_text[i]
            batch_input_ent_text_length[i, :ent_l] = batch_input_ent_cell_length[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_mask_padded[i, :ent_l, :tok_l] = batch_input_ent_mask[i][0]
            batch_input_ent_mask_padded[i, :ent_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_ent_mask[i][1]
            batch_column_entity_mask_padded[i, :col_num, :ent_l] = batch_column_entity_mask[i]
            batch_column_entity_mask_padded[i, col_num:, 0] = 1
            batch_column_header_mask_padded[i, :col_num, :tok_l] = batch_column_header_mask[i]
            batch_column_header_mask_padded[i, col_num:, 0] = 1
            batch_labels_padded[i, :col_num] = batch_labels[i]
            batch_labels_mask[i, :col_num] = batch_labels[i].sum(1)!=0
                    
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_input_ent_padded = torch.LongTensor(batch_input_ent_padded)
        batch_input_ent_text_padded = torch.LongTensor(batch_input_ent_text_padded)
        batch_input_ent_text_length = torch.LongTensor(batch_input_ent_text_length)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)

        batch_column_entity_mask_padded = torch.FloatTensor(batch_column_entity_mask_padded)
        batch_column_header_mask_padded = torch.FloatTensor(batch_column_header_mask_padded)
        batch_labels_mask = torch.FloatTensor(batch_labels_mask)
        batch_labels_padded = torch.FloatTensor(batch_labels_padded)

        return batch_table_id, batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded, \
                batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_padded, batch_input_ent_type_padded, batch_input_ent_mask_padded, \
                batch_column_entity_mask_padded, batch_column_header_mask_padded, batch_labels_mask, batch_labels_padded

class CTLoader(DataLoader):
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
    ):
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.is_train = is_train
        self.collate_fn = finetune_collate_fn_CT(dataset.tokenizer, is_train=self.is_train)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)


