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

RESERVED_HEADER_VOCAB = {0:'[PAD]',
                        1:'[MASK]'}

def process_single_header(input_data, config):
    table_id,pgTitle,secTitle,caption,headers = input_data

    tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_meta = tokenized_pgTitle+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for z in headers]
    input_tok = [config.tokenizer.convert_tokens_to_ids(config.tokenizer.mask_token)]
    input_tok_pos = [0]
    input_tok_type = [1]
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tok += list(itertools.chain(*tokenized_headers))
    input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
    input_tok_type += [1]*sum(tokenized_headers_length)
    header_mask = np.zeros([len(tokenized_headers),len(input_tok)])
    start_i = tokenized_meta_length+1
    for x in tokenized_headers_length:
        header_mask[start_i:start_i+x] = 1
        start_i += x
    input_headers = [config.header2id[x] for x in headers]

    return [table_id,np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),len(input_tok),tokenized_meta_length,np.array(input_headers),header_mask,len(tokenized_headers)]

class WikiHeaderDataset(Dataset):

    def _preprocess(self, data_dir):
        preprocessed_filename = os.path.join(
            data_dir, "procressed_HR", self.src
        )
        preprocessed_filename += ".pickle"
        if not self.force_new and os.path.exists(preprocessed_filename):
            print("try loading preprocessed data from %s" % preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)
        else:
            print("try creating preprocessed data in %s" % preprocessed_filename)
            try:
                os.mkdir(os.path.join(data_dir, "procressed_HR"))
            except FileExistsError:
                pass
            with open(os.path.join(data_dir, "{}_headers.json".format(self.src)), "r") as f:
                table_headers = json.load(f)
        print('{} {} tables'.format(len(table_headers),self.src))
        pool = Pool(processes=4)
        processed_table_headers = list(tqdm(pool.imap(partial(process_single_header,config=self), table_headers, chunksize=1000),total=len(table_headers)))
        pool.close()
        # pdb.set_trace()

        with open(os.path.join(data_dir, "procressed_HR", '{}.pickle'.format(self.src)), 'wb') as f:
            pickle.dump(processed_table_headers, f)
        # pdb.set_trace()
        return processed_table_headers

    def load_header_vocab(self, data_dir):
        header_vocab = copy.deepcopy(RESERVED_HEADER_VOCAB)
        with open(os.path.join(data_dir, "header_vocab.txt"), "r", encoding='utf8') as f:
            for line in f:
                header = line.strip()
                header_vocab[len(header_vocab)] = header
        return header_vocab

    def __init__(self, data_dir, max_input_tok=500, src="train", max_length = [50, 10], force_new=False, tokenizer = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')
        self.src = src
        self.force_new = force_new
        self.max_input_tok = max_input_tok
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.header_vocab = self.load_header_vocab(data_dir)
        self.header2id = {self.header_vocab[x]:x for x in self.header_vocab}
        self.data = self._preprocess(data_dir)
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class finetune_collate_fn_Header:
    def __init__(self, tokenizer, header_vocab_size, seed=1, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.header_vocab_size = header_vocab_size
        self.seed = seed
    def __call__(self, raw_batch):
        batch_table_id, batch_input_tok, batch_input_tok_type, batch_input_tok_pos, batch_input_tok_length, batch_tokenized_meta_length, batch_input_header, batch_header_mask, batch_input_header_length = zip(*raw_batch)

        max_input_tok_length = max(batch_input_tok_length)
        max_input_header_length = max(batch_input_header_length)
        batch_size = len(batch_input_tok_length)

        batch_input_tok_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_type_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)
        batch_input_tok_pos_padded = np.zeros([batch_size, max_input_tok_length], dtype=int)


        batch_input_mask_padded = np.zeros([batch_size, 1, max_input_tok_length], dtype=int)

        batch_seed_header = []
        batch_target_header = np.full([batch_size, self.header_vocab_size], 0, dtype=int)
        for i, (tok_l, header_l, meta_l) in enumerate(zip(batch_input_tok_length, batch_input_header_length, batch_tokenized_meta_length)):
            batch_input_tok_padded[i, :tok_l] = batch_input_tok[i]
            batch_input_tok_type_padded[i, :tok_l] = batch_input_tok_type[i]
            batch_input_tok_pos_padded[i, :tok_l] = batch_input_tok_pos[i]

            if self.seed !=-1:
                tmp_cand_header = set(range(header_l))
                # tmp_selected_header = random.sample(tmp_cand_header,self.seed)
                tmp_selected_header = list(range(self.seed))
                batch_seed_header.append(batch_input_header[i][tmp_selected_header])
                tmp_cand_header = list(tmp_cand_header-set(tmp_selected_header))
                batch_target_header[i,batch_input_header[i][tmp_cand_header]] = 1
                batch_input_mask_padded[i,:,:tok_l] = batch_header_mask[i][tmp_selected_header].sum(axis=0)
            else:
                batch_input_mask_padded[i,:,:tok_l] = 1

            batch_input_mask_padded[i, :, :meta_l+1] = 1

        
        batch_input_tok_padded = torch.LongTensor(batch_input_tok_padded)
        batch_input_tok_type_padded = torch.LongTensor(batch_input_tok_type_padded)
        batch_input_tok_pos_padded = torch.LongTensor(batch_input_tok_pos_padded)


        batch_input_mask_padded = torch.LongTensor(batch_input_mask_padded)
        batch_seed_header = torch.LongTensor(batch_seed_header)
        batch_target_header = torch.LongTensor(batch_target_header)

        return batch_table_id,batch_input_tok_padded, batch_input_tok_type_padded, batch_input_tok_pos_padded, \
                batch_input_mask_padded, batch_seed_header, batch_target_header

class WikiHeaderLoader(DataLoader):
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
        seed=1
    ):
        self.shuffle = shuffle
        if sampler is not None:
            self.shuffle = False

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.is_train = is_train
        self.collate_fn = finetune_collate_fn_Header(dataset.tokenizer, len(dataset.header_vocab), seed= seed, is_train=self.is_train)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "sampler": sampler
        }
        super().__init__(**self.init_kwargs)


