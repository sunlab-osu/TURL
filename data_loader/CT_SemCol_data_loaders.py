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


def process_single_CT(input_data, config):
    col_id, entities, entities_text, label = input_data
    input_ent = []
    input_ent_text = []
    input_ent_pos = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}
    core_entity_mask = []
    input_ent_cell_length = []
    for e_i, (index, entity) in enumerate(entities):
        tokenized_ent_text = config.tokenizer.encode(entities_text[e_i], max_length=config.max_cell_length, add_special_tokens=False)
        if len(tokenized_ent_text) == 0:
            continue
        input_ent.append(entity)
        input_ent_text.append(tokenized_ent_text)
        input_ent_cell_length.append(len(tokenized_ent_text))
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
    max_cell_length = max(input_ent_cell_length)
    input_ent_length = len(input_ent)
    input_ent_text_padded = np.zeros([input_ent_length, max_cell_length], dtype=int)
    for i,x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x
    #create input mask
    ent_ent_mask = np.eye(len(input_ent), dtype=int)
    for _,e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                    ent_ent_mask[e_i_1, e_i_2] = 1

    for _,e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                    ent_ent_mask[e_i_1, e_i_2] = 1
    return col_id, input_ent, input_ent_text_padded, input_ent_length, input_ent_cell_length, input_ent_type, input_ent_pos, ent_ent_mask, core_entity_mask, label

class SemColDataset(Dataset):

    def _preprocess(self, data_dir):
        preprocessed_filename = os.path.join(
            data_dir, "procressed_CT", self.src
        )
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                os.mkdir(os.path.join(data_dir, "procressed_CT"))
            except FileExistsError:
                pass
            with open(os.path.join(data_dir, "train_cols.json"), "r") as f:
                train_cols = json.load(f)
            class_vocab = {cls:i for i,cls in enumerate(train_cols.keys())}
            train_class_mask = np.ones(len(class_vocab))
            with open(os.path.join(data_dir, "test_cols_t2d.json"), "r") as f:
                test_cols_t2d = json.load(f)
                test_t2d_class_mask = train_class_mask
            with open(os.path.join(data_dir, "test_cols_limaye.json"), "r") as f:
                test_cols_limaye = json.load(f)
                test_limaye_class_mask = np.zeros(len(class_vocab))
                for cls in test_cols_limaye.keys():
                    test_limaye_class_mask[class_vocab[cls]] = 1
            with open(os.path.join(data_dir, "test_cols_wikipedia.json"), "r") as f:
                test_cols_wiki = json.load(f)
                test_wiki_class_mask = np.zeros(len(class_vocab))
                for cls in test_cols_wiki.keys():
                    test_wiki_class_mask[class_vocab[cls]] = 1
            def load_cached_entity(f):
                tmp = json.load(f)
                cache = {}
                for key, value in tmp.items():
                    if len(value) > 0:
                        value = value[0][28:]
                        cache[key] = self.entity_wiktitle2id.get(value, 0)
                    else:
                        cache[key] = 0
                return cache
            with open(os.path.join(data_dir, "cache_ents_T2D_Limaye.json"), "r") as f:
                cache_ents_T2D_Limaye = load_cached_entity(f)
            with open(os.path.join(data_dir, "cache_ents_Wikipedia.json"), "r") as f:
                cache_ents_Wikipedia = load_cached_entity(f)
            def load_samples(cols, sample_dir, entity_cache):
                tables = []
                for cls in cols:
                    with open(os.path.join(data_dir, sample_dir, cls+'.json'), 'r') as f:
                        cls_samples = json.load(f)
                    for col_id in cols[cls]:
                        for m_table in cls_samples[col_id]:
                            entities = []
                            entities_text = []
                            col_num = len(m_table)
                            for i, cell in enumerate(m_table['col_0']):
                                if cell != "" and cell != "NaN":
                                    entities.append([(i,0), entity_cache.get(cell, 0)])
                                    entities_text.append(cell)
                            for j in range(col_num-1):
                                for i, cell in enumerate(m_table['col_N_%d'%j]):
                                    if cell != "" and cell != "NaN":
                                        entities.append([(i,j+1), entity_cache.get(cell, 0)])
                                        entities_text.append(cell)
                            tables.append([
                                col_id,
                                entities,
                                entities_text,
                                class_vocab[cls]
                            ])
                return tables
            train_tables = load_samples(train_cols, 'train_samples',cache_ents_T2D_Limaye)
            test_tables_t2d = load_samples(test_cols_t2d, 'test_samples_t2d',cache_ents_T2D_Limaye)
            test_tables_limaye = load_samples(test_cols_limaye, 'test_samples_limaye',cache_ents_T2D_Limaye)
            test_tables_wiki = load_samples(test_cols_wiki, 'test_samples_wikipedia',cache_ents_Wikipedia)
                    
        print('{} train tables, {} test t2d tables, {} test limaye tables, {} test wikipedia tables'.format(len(train_tables),len(test_tables_t2d),len(test_tables_limaye),len(test_tables_wiki)))
        pool = Pool(processes=4)
        processed_train_tables = list(tqdm(pool.imap(partial(process_single_CT,config=self), train_tables, chunksize=1000),total=len(train_tables)))
        processed_test_tables_t2d = list(tqdm(pool.imap(partial(process_single_CT,config=self), test_tables_t2d, chunksize=1000),total=len(test_tables_t2d)))
        processed_test_tables_limaye = list(tqdm(pool.imap(partial(process_single_CT,config=self), test_tables_limaye, chunksize=1000),total=len(test_tables_limaye)))
        processed_test_tables_wiki = list(tqdm(pool.imap(partial(process_single_CT,config=self), test_tables_wiki, chunksize=1000),total=len(test_tables_wiki)))
        pool.close()
        # pdb.set_trace()

        
        with open(os.path.join(data_dir, "procressed_CT", 'train.pickle'), 'wb') as f:
            pickle.dump([class_vocab,train_class_mask,processed_train_tables], f)
        with open(os.path.join(data_dir, "procressed_CT", 'test_t2d.pickle'), 'wb') as f:
            pickle.dump([class_vocab,test_t2d_class_mask,processed_test_tables_t2d], f)
        with open(os.path.join(data_dir, "procressed_CT", 'test_limaye.pickle'), 'wb') as f:
            pickle.dump([class_vocab,test_limaye_class_mask,processed_test_tables_limaye], f)
        with open(os.path.join(data_dir, "procressed_CT", 'test_wiki.pickle'), 'wb') as f:
            pickle.dump([class_vocab,test_wiki_class_mask,processed_test_tables_wiki], f)
        # pdb.set_trace()
        if self.src == "train":
            return [class_vocab,train_class_mask,processed_train_tables]
        elif self.src == "test_t2d":
            return [class_vocab,test_t2d_class_mask,processed_test_tables_t2d]
        elif self.src == "test_limaye":
            return [class_vocab,test_limaye_class_mask,processed_test_tables_limaye]
        else:
            return [class_vocab,test_wiki_class_mask,processed_test_tables_wiki]

    def __init__(self, data_dir, entity_vocab, max_input_tok=500, src="train", max_cell_length = 10, force_new=False, tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
        self.force_new = force_new
        self.max_input_tok = max_input_tok
        self.max_cell_length = max_cell_length
        self.entity_vocab = entity_vocab
        self.entity_wiktitle2id = {self.entity_vocab[x]['wiki_title']:x for x in self.entity_vocab}
        self.class_vocab, self.class_mask, self.data = self._preprocess(data_dir)
        self.class_num = len(self.class_vocab)
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
        batch_col_id, batch_input_ent, batch_input_ent_text, batch_input_ent_length, batch_input_ent_cell_length, \
            batch_input_ent_type, batch_input_ent_pos, batch_input_ent_mask, batch_core_entity_mask, batch_labels = zip(*raw_batch)
        
        batch_size = len(batch_col_id)
        max_input_ent_length = max(batch_input_ent_length)
        max_input_cell_length = max([z.shape[-1] for z in batch_input_ent_text])

        batch_input_ent_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_text_padded = np.zeros([batch_size, max_input_ent_length, max_input_cell_length], dtype=int)
        batch_input_ent_text_length = np.ones([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_pos_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros([batch_size, max_input_ent_length, max_input_ent_length], dtype=int)
        batch_core_entity_mask_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        
        for i, ent_l in enumerate(batch_input_ent_length):
            batch_input_ent_padded[i, :ent_l] = batch_input_ent[i]
            batch_input_ent_text_padded[i, :ent_l, :batch_input_ent_text[i].shape[-1]] = batch_input_ent_text[i]
            batch_input_ent_text_length[i, :ent_l] = batch_input_ent_cell_length[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_pos_padded[i, :ent_l] = batch_input_ent_pos[i]
            batch_input_ent_mask_padded[i, :ent_l, :ent_l] = batch_input_ent_mask[i]
            batch_core_entity_mask_padded[i, :ent_l] = batch_core_entity_mask[i]
        
        batch_input_ent_padded = torch.LongTensor(batch_input_ent_padded)
        batch_input_ent_text_padded = torch.LongTensor(batch_input_ent_text_padded)
        batch_input_ent_text_length = torch.LongTensor(batch_input_ent_text_length)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_pos_padded = torch.LongTensor(batch_input_ent_pos_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)
        batch_core_entity_mask_padded = torch.BoolTensor(batch_core_entity_mask_padded)
        batch_input_ent_mask_padded *= batch_core_entity_mask_padded[:,:,None]
        batch_labels = torch.LongTensor(batch_labels)

        return batch_col_id, batch_input_ent_padded, batch_input_ent_text_padded, batch_input_ent_text_length, \
            batch_input_ent_type_padded, batch_input_ent_pos_padded, batch_input_ent_mask_padded, batch_core_entity_mask_padded, batch_labels

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


