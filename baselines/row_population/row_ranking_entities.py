"""
Estimate P(E|e_i+1) of ranking entities in row population

author: Shuo Zhang
"""

from elastic import Elastic
from row_evaluation import Row_evaluation
from scorer import ScorerLM
import math
import json
import re
import pdb
from tqdm import tqdm
import numpy as np
import os
import pickle
from metric import mean_average_precision

class P_e_e(Row_evaluation):
    def __init__(self, type_index_name="wikipedia_category",table_index_name="table_index_wikitable_train_jan_13",abstract_index_name="dbpedia_2015_10_abstract", lamda=0.5):
        """

        :param index_name: name of index
        :param lamda: smoothing parameter
        """
        super().__init__(type_index_name, table_index_name)
        self.__lambda = lamda
        self.index_name = table_index_name
        self.__tes = Elastic(table_index_name)
        self.__elas = Elastic(abstract_index_name)
        self.__mu = 0.5
        self.e_c_freq = pickle.load(open("./data/entity_caption_term_freq.pkl", "rb"))
        self.e_ht_freq = pickle.load(open("./data/entity_header_term_freq.pkl", "rb"))
        self.e_h_freq = pickle.load(open("./data/entity_header_freq.pkl", "rb"))
        self.c_freq = pickle.load(open("./data/caption_term_freq.pkl", "rb"))
        self.ht_freq = pickle.load(open("./data/header_term_freq.pkl", "rb"))
        self.h_freq = pickle.load(open("./data/header_freq.pkl", "rb"))
        self.e_t = pickle.load(open("./data/entity_tables.pkl", "rb"))

    def rank_candidates(self, seed, c=None, l=None):
        cand = list(self.find_candidates_e(seed_E=seed)) + list(self.find_candidates_c(seed_E=seed, c=c)) + list(
            self.find_candidates_cat(seed_E=seed))
        p_all = {}
        pee = self.estimate_pee(cand, seed)
        pce = self.estimate_pce(cand, c)
        ple = self.estimate_ple(cand, l)
        for entity, score in pee.items():
            p_all[entity] = max(0.000001, score) * max(0.000001, pce.get(entity)) * max(0.000001, ple.get(entity))
        return p_all

    def rank_core_candidates(self, seed, c=None, l=None, num=100):
        # pdb.set_trace()
        cand_e = self.find_core_candidates_e(seed_E=seed, num=num)
        cand_c = self.find_core_candidates_c(seed_E=seed, c=c, num=num)
        cand = cand_e | cand_c
        cand = list(cand)
        p_all = {}
        pee = self.estimate_pee_fast(cand, seed)
        pce = self.estimate_pce_nokb_fast(cand, c)
        ple = self.estimate_ple_fast(cand, l)
        # pdb.set_trace()
        for entity, score in pee.items():
            p_all[entity] = max(0.000001, score) * max(0.000001, pce.get(entity)) * max(0.000001, ple.get(entity))
        return p_all, pee, pce, ple, cand_e, cand_c

    def estimate_pee_fast(self, cand, seed):
        """Estimate P(c|e_i+1) for candidates"""
        # pdb.set_trace()
        p_all = {}
        n_e = self.get_tnum_contain_seed_fast(seed)
        for entity in cand:
            n_e_i = self.get_tnum_contain_seed_fast(entity)  # number of tables containing e_i+1
            seed_e = []
            seed_e.append(entity)
            for en in seed:
                seed_e.append(en)
            n_e_e = self.get_tnum_contain_seed_fast(seed_e)  # number of tables containing e_i+1 and E
            sim = 0  # todo
            if n_e_i == 0:
                p_all[entity] = 0
            elif n_e == 0:
                p_all[entity] = (1 - self.__lambda) * sim  # /n_e_i
            else:
                p_all[entity] = ((self.__lambda * (n_e_e / n_e) + (1 - self.__lambda) * sim))  # /n_e_i
        return p_all
    
    def get_tnum_contain_seed_fast(self, seed):
        if not isinstance(seed, list):
            return len(self.e_t.get(seed, []))
        elif len(seed) == 1:
            return len(self.e_t.get(seed[0],[]))
        else:
            current_set = set(self.e_t.get(seed[0],[]))
            for entity in seed[1:]:
                current_set = current_set & set(self.e_t.get(entity, []))
            return len(current_set)

    def generate_search_body_multi(self, seed):
        """Generate and return search body"""
        body = {}
        if len(seed) == 1:  # One constraints
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "match": {"core_entity_n": seed[0]}
                        }
                    }
                }
            }
        else:  # Multiple constraints
            must = []
            must.append({"match": {"core_entity_n": seed[0]}})
            for item in seed[1:]:
                must.append({"match": {"core_entity_n": item}})
            body = {
                "query": {
                    "bool": {
                        "must": must
                    }
                }
            }
        return body

    def estimate_pce(self, cand, c):
        """Estimate P(c|e_i+1) for candidates"""
        p_all = {}
        caption = self.parse(c)  # Put query into a list
        for entity_id in cand:
            p = 0
            body = self.generate_search_body(entity_id, field="entity")
            table_ids = self.__tes.search_complex(body).keys()  # Search table containing entity

            kb_l = self.__elas.doc_length(entity_id, "abstract")  # entity abstract length
            kb_c_l = self.__elas.coll_length("abstract")  # entity abstract collection length
            collection_l = self.__tes.coll_length("caption")  # caption collection length
            for t in caption:  # Iterate term in caption
                term = self.__tes.analyze_query(t)
                c_l, tf = 0, 0  # caption length, term freq
                for table_id in table_ids:
                    c_l += self.__tes.doc_length(table_id, "caption")  # caption length
                    tf += self.__tes.term_freq(table_id, "caption", term)  # caption term frequency
                tf_c = self.__tes.coll_term_freq(term, "caption")
                kb_tf = self.__elas.term_freq(entity_id, "abstract", term)  # n(t,kb)
                kb_c_tf = self.__elas.coll_term_freq(term, "abstract")  # term freq in kb collection
                p += self.estimate_p(kb_l, kb_tf, kb_c_l, kb_c_tf, tf, c_l, tf_c, collection_l)
            if p != 0:
                p = math.exp(p)
            p_all[entity_id] = p
        return p_all

    def estimate_p(self, kb_l, kb_tf, kb_c_l, kb_c_tf, tf, c_l, tf_c, collection_l):
        """P(t_c|e_i+1)"""
        p_kb = self.__lambda * (kb_tf + self.__mu * kb_c_tf / kb_c_l) / (kb_l + self.__mu) + (1 - self.__lambda) * (
            tf + self.__mu * tf_c / collection_l) / (c_l + self.__mu)
        if p_kb != 0:
            p_kb = math.log(p_kb)
        return p_kb

    def estimate_pce_nokb(self, cand, c, cache={}):
        """Estimate P(c|e_i+1) for candidates"""
        # pdb.set_trace()
        p_all = {}
        caption = self.parse(c)  # Put query into a list
        table_term_cache = {}
        collection_l = self.__tes.coll_length("caption")
        for entity_id in cand:
            p = 0
            try:
                table_ids = cache[entity_id]
            except:
                body = self.generate_search_body(entity_id, field="core_entity_n")
                table_ids = self.__tes.search_complex(body).keys()  # Search table containing entity
              # caption collection length
            for t in caption:  # Iterate term in caption
                term = self.__tes.analyze_query(t)
                c_l, tf = 0, 0  # caption length, term freq
                for table_id in table_ids:
                    if table_id not in table_term_cache:
                        table_term_cache[table_id] = [self.__tes.term_freqs(table_id, "caption"), self.__tes.doc_length(table_id, "caption")]
                    c_l += table_term_cache[table_id][1]  # caption length
                    tf += table_term_cache[table_id][0].get(term, 0)  # caption term frequency
                tf_c = self.__tes.coll_term_freq(term, "caption")
                p += self.estimate_p_nokb(tf, c_l, tf_c, collection_l)
            if p != 0:
                p = math.exp(p)
            p_all[entity_id] = p
        return p_all

    def estimate_pce_nokb_fast(self, cand, c):
        """Estimate P(c|e_i+1) for candidates"""
        # pdb.set_trace()
        p_all = {}
        caption = self.parse(c)  # Put query into a list
        collection_l = self.c_freq[0]
        for entity_id in cand:
            p = 0
            c_l = self.e_c_freq.get(entity_id, [0,{}])[0]
            for t in caption:  # Iterate term in caption
                tf = self.e_c_freq.get(entity_id, [0,{}])[1].get(t, 0)
                tf_c = self.c_freq[1].get(t, 0)
                p += self.estimate_p_nokb(tf, c_l, tf_c, collection_l)
            if p != 0:
                p = math.exp(p)
            p_all[entity_id] = p
        return p_all

    def estimate_p_nokb(self, tf, c_l, tf_c, collection_l):
        """P(t_c|e_i+1)"""
        p_kb = (tf + self.__mu * tf_c / collection_l) / (c_l + self.__mu)
        if p_kb != 0:
            p_kb = math.log(p_kb)
        return p_kb

    def estimate_ple(self, cand, l, cache={}):
        """Estimate P(l|e_i+1) for candidates"""
        # pdb.set_trace()
        p_all = {}
        table_heading_cache = {}
        for entity in cand:
            p_all[entity] = 0
            for label in l:
                try:
                    n_e = len(cache[entity])
                except:
                    body = self.generate_search_body([entity], field="core_entity_n")
                    n_e = self.__tes.estimate_number_complex(body)  # number of tables containing e_i+1
                body2 = self.generate_search_body_l([entity, label])
                n_l_e = self.__tes.estimate_number_complex(body2)  # number of tables containing e_i+1&label

                table_ids = self.__tes.get_ids(body2)
                if table_ids != 0:
                    p_l_theta = self.p_l_theta_lm(label, table_ids, cache=table_heading_cache)
                    if n_e == 0:
                        p_all[entity] += self.__lambda * p_l_theta
                    else:
                        p_all[entity] += self.__lambda * p_l_theta + (1 - self.__lambda) / len(l) * n_l_e / n_e
        return p_all

    def estimate_ple_fast(self, cand, l):
        """Estimate P(l|e_i+1) for candidates"""
        # pdb.set_trace()
        p_all = {}
        for entity in cand:
            p_all[entity] = 0
            for label in l:
                n_e = self.e_h_freq.get(entity, [0,{}])[0]  # number of tables containing e_i+1
                n_l_e = self.e_h_freq.get(entity, [0,{}])[1].get(label, 0)  # number of tables containing e_i+1&label

                table_ids = self.e_t[entity]
                if table_ids != 0:
                    p_l_theta = self.p_l_theta_lm_fast(label, entity)
                    if n_e == 0:
                        p_all[entity] += self.__lambda * p_l_theta
                    else:
                        p_all[entity] += self.__lambda * p_l_theta + (1 - self.__lambda) / len(l) * n_l_e / n_e
        return p_all

    def p_l_theta_lm(self, label, table_ids, cache={}):
        # pdb.set_trace()
        """Using language modeling estimate P(l|theta)"""
        p_label = self.parse(label)
        p_l_theta = 0
        c_l = self.__tes.coll_length("headings")  # collection length
        for t in p_label:
            a_t = self.__tes.analyze_query(t)
            l_l = 0  # table label length(table containing t)
            t_f = 0  # tf of label
            c_tf = self.__tes.coll_term_freq(a_t, "headings")  # tf in collection
            for table_id in table_ids:
                if table_id not in cache:
                    cache[table_id] = self.__tes.term_freqs(table_id, "headings")
                l_l += self.__tes.doc_length(table_id, "headings")
                t_f += cache[table_id].get(a_t, 0)
            if l_l + self.__mu != 0:
                p = (t_f + self.__mu * c_tf / c_l) / (l_l + self.__mu)
            else:
                p = 0
            if p != 0:
                p_l_theta += math.log(p)
            else:
                p_l_theta += 0

        if p_l_theta != 0:
            p_l_theta = math.exp(p_l_theta)
        return p_l_theta

    def p_l_theta_lm_fast(self, label, entity_id):
        # pdb.set_trace()
        """Using language modeling estimate P(l|theta)"""
        p_label = self.parse(label)
        p_l_theta = 0
        c_l = self.ht_freq[0]  # collection length
        for t in p_label:
            l_l = self.e_ht_freq.get(entity_id, [0,{}])[0]  # table label length(table containing t)
            t_f = self.e_ht_freq.get(entity_id, [0,{}])[1].get(t, 0)  # tf of label
            c_tf = self.ht_freq[1].get(t, 0)  # tf in collection
            if l_l + self.__mu != 0:
                p = (t_f + self.__mu * c_tf / c_l) / (l_l + self.__mu)
            else:
                p = 0
            if p != 0:
                p_l_theta += math.log(p)

        if p_l_theta != 0:
            p_l_theta = math.exp(p_l_theta)
        return p_l_theta

    def generate_search_body_l(self, query):
        """Generate and return search body"""
        body = {}
        if len(query) == 1:
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "term": {"core_entity_n": query[0]}
                        }
                    }
                }
            }
        elif len(query) == 2:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {"core_entity_n": query[0]}
                            },
                            {
                                "match_phrase": {"headings": query[1]}
                            }
                        ]
                    }
                }}
        return body

def parse(h):
    """entity [A|B]----B"""
    if "[" in h and "|" in h and "]" in h:
        return h.split("|")[1].split("]")[0]
    else:
        return h


def label_replace(headings):
    """Only keep entity strings"""
    return [parse(i) for i in headings]

def load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=1):
    entity_vocab = {}
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
                    'count': count
                }
    print('total number of entity: %d\nremove because of empty title: %d\nremove because count<%d: %d'%(len(entity_vocab),bad_title,min_ent_count,few_entity))
    return entity_vocab

if __name__ == "__main__":
    seed_num = 1
    k = 50
    eva = P_e_e()
    dev_result = {}
    data_dir = "./data"
    entity_vocab = load_entity_vocab(data_dir,True, min_ent_count=2)
    all_entity_set = set([item['wiki_id'] for _,item in entity_vocab.items()])
    tables_ignored = 0
    with open(os.path.join(data_dir,"dev_tables.jsonl"), 'r') as f:
        for line in tqdm(f):
            table = json.loads(line.strip())
            table_id = table.get("_id", "")
            pgTitle = table.get("pgTitle", "").lower()
            secTitle = table.get("sectionTitle", "").lower()
            caption = table.get("tableCaption", "").lower()
            headers = table.get("processed_tableHeaders", [])
            rows = table.get("tableData", {})
            entity_columns = table.get("entityColumn", [])
            headers = [headers[j] for j in entity_columns]
            entity_cells = np.array(table.get("entityCell",[[]]))
            core_entities = []
            num_rows = len(rows)
            entities = []

            for i in range(num_rows):
                for j in entity_columns:
                    if entity_cells[i,j] == 1:
                        entity = rows[i][j]['surfaceLinks'][0]['target']['id']
                        if entity == "":
                            continue
                        entities.append(entity)
                        if j == 0:
                            core_entities.append(entity)
            catcallall = " ".join([pgTitle, secTitle, caption, " ".join(headers)])
            remained_core_entities = [z for z in core_entities if z in all_entity_set]
            if len(remained_core_entities) < 5:
                tables_ignored += 1
                continue
            seed = remained_core_entities[:1]

            # pdb.set_trace()
            # A1 = eva.find_core_candidates_cat(seed, k)
            # A1 = set()
            # B = eva.find_core_candidates_c(seed, re.escape(catcallall), k)
            # C = eva.find_core_candidates_e(seed, k)
            # pdb.set_trace()
            pall, pee, pce, ple, cand_e, cand_c = eva.rank_core_candidates(seed, re.escape(caption), [re.escape(headers[0])], num=k)
            target_entities = set(remained_core_entities[1:])
            ranked_entities = [1 if z[0] in target_entities else 0 for z in sorted(pall.items(),key=lambda z:z[1],reverse=True)]
            # dev_result[table_id] = [set(seed), B, C, B|C]
            dev_result[table_id] = [set(seed), target_entities, ranked_entities, pall, pee, pce, ple, cand_e, cand_c]
            # pdb.set_trace()
    # pdb.set_trace()
    # for i in range(3):
        # print(np.mean([len(x[0]&x[i+1])/len(x[0]) for _,x in dev_result.items()]), np.mean([len(x[i+1])for _,x in dev_result.items()]))
    print("tables ignored %d"%tables_ignored)
    pdb.set_trace()
    print("map: %f"%mean_average_precision([z[2] for _,z in dev_result.items()]))
    with open(os.path.join(data_dir, "dev_result.pkl"),"wb") as f:
        pickle.dump(dev_result, f)
    print("finish val")
        