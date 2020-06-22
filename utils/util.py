import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, Counter
import pickle
import copy
import os

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

def create_ent_embedding(data_dir, ent_vocab, origin_embed):
    with open(os.path.join(data_dir, 'entity_embedding_tinybert_312.pkl'), 'rb') as f:
        ent_embed = pickle.load(f)
    for wiki_id in ent_vocab:
        origin_embed[ent_vocab[wiki_id]] = ent_embed[str(wiki_id)]
    return origin_embed

def create_header_embedding(data_dir, header_vocab, origin_embed, is_bert=False):
    with open(os.path.join(data_dir, 'header_embedding_312_bert.pkl' if is_bert else 'header_embedding_312.pkl'), 'rb') as f:
        header_embed = pickle.load(f)
    for header_id in header_vocab:
        origin_embed[header_id] = header_embed[header_vocab[header_id]]
    return origin_embed

RESERVED_ENT_VOCAB = {0:{'wiki_id':'[PAD]', 'wiki_title':'[PAD]', 'count': -1, 'mid': -1},
                        1:{'wiki_id':'[ENT_MASK]','wiki_title':'[ENT_MASK]', 'count': -1, 'mid': -1},
                        2:{'wiki_id':'[PG_ENT_MASK]','wiki_title':'[PG_ENT_MASK]', 'count': -1, 'mid': -1},
                        3:{'wiki_id':'[CORE_ENT_MASK]','wiki_title':'[CORE_ENT_MASK]', 'count': -1, 'mid': -1}
                        }
RESERVED_ENT_VOCAB_NUM = len(RESERVED_ENT_VOCAB)

def load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=1):
    entity_vocab = copy.deepcopy(RESERVED_ENT_VOCAB)
    bad_title = 0
    few_entity = 0
    with open(os.path.join(data_dir, 'entity_vocab.txt'), 'r', encoding="utf-8") as f:
        for line in f:
            _, entity_id, entity_title, entity_mid, count = line.strip().split('\t')
            if ignore_bad_title and entity_title == '':
                bad_title += 1
            elif int(count) < min_ent_count:
                few_entity += 1
            else:
                entity_vocab[len(entity_vocab)] = {
                    'wiki_id': int(entity_id),
                    'wiki_title': entity_title,
                    'mid': entity_mid,
                    'count': int(count)
                }
    print('total number of entity: %d\nremove because of empty title: %d\nremove because count<%d: %d'%(len(entity_vocab),bad_title,min_ent_count,few_entity))
    return entity_vocab

def generate_vocab_distribution(entity_vocab):
    distribution = np.zeros(len(entity_vocab))
    for i, item in entity_vocab.items():
        if i in RESERVED_ENT_VOCAB:
            distribution[i] = 2
        else:
            distribution[i] = int(item['count'])
    distribution = np.log10(distribution)
    distribution /= np.sum(distribution)
    return distribution

def load_type_vocab(data_dir):
    type_vocab = {}
    with open(os.path.join(data_dir, "type_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            type_vocab[t] = int(index)
    return type_vocab

def load_relation_vocab(data_dir):
    relation_vocab = {}
    with open(os.path.join(data_dir, "relation_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            relation_vocab[t] = int(index)
    return relation_vocab

def load_dbpedia_type_vocab(data_dir):
    type_vocab = {}
    with open(os.path.join(data_dir, "dbpedia_type_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            type_vocab[t] = int(index)
    return type_vocab