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


def process_single_EL(input_data, config):
    table_id, pgTitle, secTitle, caption, headers, entities, candidate_entities, labels,_ = input_data

    cand_name = []
    cand_description = []
    cand_type = []
    for name, description, types in candidate_entities:
        if name is None:
            name=''
        if description is None:
            description=''
        if types is None:
            types=[]
        cand_name.append(config.tokenizer.encode(name, max_length=config.max_cell_length, add_special_tokens=False))
        cand_description.append(config.tokenizer.encode(description, max_length=config.max_description_length, add_special_tokens=False))
        cand_type.append([config.ent_type_vocab.get(t,0) for t in types])
    cand_name_length = [len(x) for x in cand_name]
    cand_description_length = [len(x) for x in cand_description]
    cand_type_length = [len(x) for x in cand_type]
    max_cand_name_length = max(cand_name_length)
    max_cand_description_length = max(cand_description_length)
    max_cand_type_length = max(cand_type_length)
    cand_name_padded = np.zeros([len(cand_name), max_cand_name_length], dtype=int)
    cand_description_padded = np.zeros([len(cand_description), max_cand_description_length], dtype=int)
    cand_type_padded = np.zeros([len(cand_type), max_cand_type_length], dtype=int)
    for i,(x,y,z) in enumerate(zip(cand_name_length,cand_description_length,cand_type_length)):
        cand_name_padded[i,:x] = cand_name[i]
        cand_description_padded[i,:y] = cand_description[i]
        cand_type_padded[i,:z] = cand_type[i]

    # pgEnt = config.entity_wikid2id.get(pgEnt, -1)
    pgEnt = -1

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

    input_ent_text = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}
    entities_index = []
    for e_i, (index, cell) in enumerate(entities):
        entities_index.append(index)
        entity_text = cell
        tokenized_ent_text = config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False)
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
    input_ent_length = len(input_ent_text)
    #create input mask
    tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
    meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent_text)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent_text)], dtype=int)
    start_i = 0
    header_span = {}
    for h_i, _ in enumerate(headers):
        header_span[h_i] = (start_i, start_i+tokenized_headers_length[h_i])
        start_i += tokenized_headers_length[h_i]
    for e_i, (index, _) in enumerate(entities):
        header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    ent_header_mask = np.transpose(header_ent_mask)

    input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
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
    # prepend pgEnt to input_ent, input_ent[0] = pgEnt
    if pgEnt!=-1:
        input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    else:
        input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int),input_tok_mask[1]],axis=1)
    input_ent_text = [tokenized_pgTitle[:config.max_cell_length]] + input_ent_text
    input_ent_type = [2] + input_ent_type

    new_input_ent_mask = [np.ones([len(input_ent_text), len(input_tok)], dtype=int), np.ones([len(input_ent_text), len(input_ent_text)], dtype=int)]
    new_input_ent_mask[0][1:, :] = input_ent_mask[0]
    new_input_ent_mask[1][1:, 1:] = input_ent_mask[1]
    if pgEnt==-1:
        new_input_ent_mask[1][:, 0] = 0
        new_input_ent_mask[1][0, :] = 0

    input_ent_mask = new_input_ent_mask
    input_ent_cell_length = [len(x) if len(x)!=0 else 1 for x in input_ent_text]
    max_cell_length = max(input_ent_cell_length)
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    for i,x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x

    return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),(np.array(input_tok_mask[0]),np.array(input_tok_mask[1])),len(input_tok), \
                input_ent_text_padded,input_ent_cell_length,np.array(input_ent_type),(np.array(input_ent_mask[0]),np.array(input_ent_mask[1])),len(input_ent_text), \
                cand_name_padded,cand_name_length,cand_description_padded,cand_description_length,cand_type_padded,cand_type_length,len(cand_name_length),np.array(labels),entities_index]

class ELDataset(Dataset):

    def _preprocess(self, data_dir):
        preprocessed_filename = os.path.join(
            data_dir, "procressed_EL", self.src
        )
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                os.mkdir(os.path.join(data_dir, "procressed_EL"))
            except FileExistsError:
                pass
            with open(os.path.join(data_dir, "{}.table_entity_linking.json".format(self.src)), "r") as f:
                data = json.load(f)

        print('{} {} tables'.format(len(data),self.src))
        process_single_EL(data[0], self)
        pool = Pool(processes=10)
        processed_data = list(tqdm(pool.imap(partial(process_single_EL,config=self), data, chunksize=1000),total=len(data)))
        pool.close()
        # pdb.set_trace()

        
        with open(os.path.join(data_dir, "procressed_EL", '{}.pickle'.format(self.src)), 'wb') as f:
            pickle.dump(processed_data, f)
        # pdb.set_trace()
        return processed_data

    def __init__(self, data_dir, ent_type_vocab, max_input_tok=500, src="train", max_length = [50, 10, 10, 100], force_new=False, tokenizer = None):
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
        self.max_description_length = max_length[3]
        self.ent_type_vocab = ent_type_vocab
        self.ent_type_num = len(self.ent_type_vocab)
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class finetune_collate_fn_EL:
    def __init__(self, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
    def __call__(self, raw_batch):
        batch_table_id, batch_input_tok, batch_input_tok_type, batch_input_tok_pos, batch_input_tok_mask, batch_input_tok_length, \
            batch_input_ent_text, batch_input_ent_cell_length, batch_input_ent_type, batch_input_ent_mask, batch_input_ent_length, \
            batch_cand_name,batch_cand_name_length_tmp,batch_cand_description,batch_cand_description_length_tmp,batch_cand_type,batch_cand_type_length_tmp,batch_cand_length,batch_labels,batch_entities_index = zip(*raw_batch)
        
        batch_size = len(batch_table_id)
        max_input_tok_length = max(batch_input_tok_length)
        max_input_ent_length = max(batch_input_ent_length)
        max_input_cell_length = max([z.shape[-1] for z in batch_input_ent_text])
        max_cand_name_length = max([z.shape[-1] for z in batch_cand_name])
        max_cand_description_length = max([z.shape[-1] for z in batch_cand_description])
        max_cand_type_length = max([z.shape[-1] for z in batch_cand_type])
        max_cand_size = max(batch_cand_length)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_mask_padded = np.zeros([batch_size, max_input_tok_length, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_input_ent_text_padded = np.zeros([batch_size, max_input_ent_length, max_input_cell_length], dtype=int)
        batch_input_ent_text_length = np.ones([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_type_padded = np.zeros([batch_size, max_input_ent_length], dtype=int)
        batch_input_ent_mask_padded = np.zeros([batch_size, max_input_ent_length, max_input_tok_length+max_input_ent_length], dtype=int)

        batch_cand_name_padded = np.zeros([batch_size, max_cand_size, max_cand_name_length], dtype=int)
        batch_cand_name_length = np.ones([batch_size, max_cand_size], dtype=int)
        batch_cand_description_padded = np.zeros([batch_size, max_cand_size, max_cand_description_length], dtype=int)
        batch_cand_description_length = np.ones([batch_size, max_cand_size], dtype=int)
        batch_cand_type_padded = np.zeros([batch_size, max_cand_size, max_cand_type_length], dtype=int)
        batch_cand_type_length = np.ones([batch_size, max_cand_size], dtype=int)
        batch_cand_mask_padded = np.zeros([batch_size, max_cand_size], dtype=int)

        
        batch_labels_padded = np.full([batch_size, max_input_ent_length-1], -1, dtype=int)
        # pdb.set_trace()
        for i, (tok_l, ent_l, cand_l) in enumerate(zip(batch_input_tok_length, batch_input_ent_length, batch_cand_length)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]
            batch_input_tok_mask_padded[i, :tok_l, :tok_l] = batch_input_tok_mask[i][0]
            batch_input_tok_mask_padded[i, :tok_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_tok_mask[i][1]

            batch_cand_name_padded[i, :cand_l, :batch_cand_name[i].shape[-1]] = batch_cand_name[i]
            batch_cand_name_length[i, :cand_l] = batch_cand_name_length_tmp[i]
            batch_cand_description_padded[i, :cand_l, :batch_cand_description[i].shape[-1]] = batch_cand_description[i]
            batch_cand_description_length[i, :cand_l] = batch_cand_description_length_tmp[i]
            batch_cand_type_padded[i, :cand_l, :batch_cand_type[i].shape[-1]] = batch_cand_type[i]
            batch_cand_type_length[i, :cand_l] = batch_cand_type_length_tmp[i]
            batch_cand_mask_padded[i, :cand_l] = 1

            batch_input_ent_text_padded[i, :ent_l, :batch_input_ent_text[i].shape[-1]] = batch_input_ent_text[i]
            batch_input_ent_text_length[i, :ent_l] = batch_input_ent_cell_length[i]
            batch_input_ent_type_padded[i, :ent_l] = batch_input_ent_type[i]
            batch_input_ent_mask_padded[i, :ent_l, :tok_l] = batch_input_ent_mask[i][0]
            batch_input_ent_mask_padded[i, :ent_l, max_input_tok_length:max_input_tok_length+ent_l] = batch_input_ent_mask[i][1]
            batch_labels_padded[i, :ent_l-1] = batch_labels[i]
        
        batch_cand_name_length[batch_cand_name_length==0] = 1
        batch_cand_description_length[batch_cand_description_length==0] = 1
        batch_cand_type_length[batch_cand_type_length==0] = 1
                    
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)
        batch_input_tok_mask_padded = torch.LongTensor(batch_input_tok_mask_padded)

        batch_cand_name_padded = torch.LongTensor(batch_cand_name_padded)
        batch_cand_name_length = torch.LongTensor(batch_cand_name_length)
        batch_cand_description_padded = torch.LongTensor(batch_cand_description_padded)
        batch_cand_description_length = torch.LongTensor(batch_cand_description_length)
        batch_cand_type_padded = torch.LongTensor(batch_cand_type_padded)
        batch_cand_type_length = torch.LongTensor(batch_cand_type_length)
        batch_cand_mask_padded = torch.LongTensor(batch_cand_mask_padded)

        batch_input_ent_text_padded = torch.LongTensor(batch_input_ent_text_padded)
        batch_input_ent_text_length = torch.LongTensor(batch_input_ent_text_length)
        batch_input_ent_type_padded = torch.LongTensor(batch_input_ent_type_padded)
        batch_input_ent_mask_padded = torch.LongTensor(batch_input_ent_mask_padded)

        batch_labels_padded = torch.LongTensor(batch_labels_padded)

        return batch_table_id, batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, batch_input_tok_mask_padded, \
                batch_input_ent_text_padded, batch_input_ent_text_length, batch_input_ent_type_padded, batch_input_ent_mask_padded, \
                batch_cand_name_padded, batch_cand_name_length,batch_cand_description_padded, batch_cand_description_length,batch_cand_type_padded, batch_cand_type_length, batch_cand_mask_padded, \
                batch_labels_padded,batch_entities_index

class ELLoader(DataLoader):
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
        self.collate_fn = finetune_collate_fn_EL(dataset.tokenizer, is_train=self.is_train)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)


