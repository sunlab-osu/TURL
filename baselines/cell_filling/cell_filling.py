import numpy as np
import os
import json
import pickle
from gensim.models import KeyedVectors

import pdb
from tqdm import tqdm

class cell_filling(object):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'e2e_row.json'), 'r') as f:
            self.e2e_row = json.load(f)
            self.e2e_row = {int(key):value for key,value in self.e2e_row.items()}
        with open(os.path.join(data_dir, "e2column.json"),"r") as f:
            self.e2column = json.load(f)
            self.e2column = {int(key):value for key,value in self.e2column.items()}
        with open(os.path.join(data_dir, "table_column2e"),"r") as f:
            self.table_column2e = json.load(f)
        with open(os.path.join(data_dir, "n_h2h.pkl"),"rb") as f:
            self.n_h2h = pickle.load(f)
        self.header_vectors = KeyedVectors.load(os.path.join(data_dir, "header_vectors.kv"), mmap='r')

    
    def get_cand_row(self, seed, h=None):
        cands = {}
        for t_id,c_id,c_name,e in self.e2e_row.get(seed, []):
            if h is None or h not in self.n_h2h or c_name in self.n_h2h[h] or c_name==h:
                if e not in cands:
                    cands[e] = [set(),set()]
                cands[e][0].add(c_name)
                cands[e][1].add((t_id, c_id))
        return cands
    
    def get_cand_row_relax(self, seed, h=None):
        cands = {}
        for t_id,c_id,c_name,e in self.e2e_row.get(seed, []):
            if e not in cands:
                cands[e] = [set(),set()]
            cands[e][0].add(c_name)
            cands[e][1].add((t_id, c_id))
        return cands

    def rank_cand_exact(self, h, cands):
        cand_h_scores = []
        # h_seen = h in self.header_vectors.vocab
        h_seen = h in self.n_h2h
        for e in cands:
            tmp = []
            for z in cands[e][0]:
                if z == h:
                    tmp.append(10000)
                else:
                    tmp.append(-1)
                # elif not h_seen:
                #     tmp.append(-1)
                # else:
                #     try:
                #         # tmp.append(self.header_vectors.similarity(h, z))
                #         tmp.append(self.n_h2h[h][z])
                #     except:
                #         tmp.append(-1)
            score = max(tmp)
            cand_h_scores.append((e, score))
        sorted_cands = sorted(cand_h_scores, key=lambda z:z[1], reverse=True)
        return [z[0] for z in sorted_cands]
    
    def rank_cand_h2h(self, h, cands):
        cand_h_scores = []
        h_seen = h in self.n_h2h
        for e in cands:
            tmp = []
            for z in cands[e][0]:
                if z == h:
                    tmp.append(10000)
                # else:
                #     tmp.append(-1)
                elif not h_seen:
                    tmp.append(-1)
                else:
                    try:
                        tmp.append(self.n_h2h[h][z])
                    except:
                        tmp.append(-1)
            score = max(tmp)
            cand_h_scores.append((e, score))
        sorted_cands = sorted(cand_h_scores, key=lambda z:z[1], reverse=True)
        return [z[0] for z in sorted_cands]

    def rank_cand_h2v(self, h, cands):
        cand_h_scores = []
        h_seen = h in self.header_vectors.vocab
        for e in cands:
            tmp = []
            for z in cands[e][0]:
                if z == h:
                    tmp.append(10000)
                # else:
                #     tmp.append(-1)
                elif not h_seen:
                    tmp.append(-1)
                else:
                    try:
                        tmp.append(self.header_vectors.similarity(h, z))
                    except:
                        tmp.append(-1)
            score = max(tmp)
            cand_h_scores.append((e, score))
        sorted_cands = sorted(cand_h_scores, key=lambda z:z[1], reverse=True)
        return [z[0] for z in sorted_cands]
    
    def get_cand_column(self, seeds):
        cands = {}
        for seed in seeds:
            for t_id, c_id in self.e2column[seed]:
                for e in self.table_column2e[t_id+'-'+c_id]:
                    if e not in cands:
                        cands[e] = [set(),set()]
                    cands[e][0].add(seed)
                    cands[e][1].add((t_id, c_id))
        return cands             

if __name__ == "__main__":
    data_dir = "./data"
    CF = cell_filling(data_dir)
    with open(os.path.join(data_dir,"CF_dev_data.json"), 'r') as f:
        dev_data = json.load(f)
    recall = []
    precision = []
    overall_precision = []
    for table_id,pgEnt,pgTitle,secTitle,caption,(h1, h2),data_sample in tqdm(dev_data):
        r = 0.0
        p = 0.0
        for core_e, target_e in data_sample:
            cands = CF.get_cand_row(core_e, h2)
            if target_e in cands:
                r += 1
                sorted_cands = CF.rank_cand_h(h2, cands)
                if target_e in sorted_cands[:5]:
                    p+=1
            # pdb.set_trace()
        recall.append(r/len(data_sample))
        if r!=0:
            precision.append(p/r)
        overall_precision.append(p/len(data_sample))
    print("recall: %f"%np.mean(recall))
    print("precision: %f"%np.mean(precision))
    print("overall precision: %f"%np.mean(overall_precision))