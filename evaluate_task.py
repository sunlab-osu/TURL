from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from data_loader.hybrid_data_loaders import *
from model.configuration import TableConfig
from model.model import HybridTableMaskedLM, HybridTableCER
from model.transformers import BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'CER': (TableConfig, HybridTableCER, BertTokenizer),
    'CF' : (TableConfig, HybridTableMaskedLM, BertTokenizer)
}


def evaluate_CER(args, dataset, seed_num=1, debug=False):
    config_class, model_class, _ = MODEL_CLASSES['CER']
    config = config_class.from_pretrained(args.config_name)
    config.output_attentions = True
    
    model = model_class(config, is_simple=True)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    all_entity_set = set(dataset.entity_wikid2id.keys())
    tables_ignored = 0
    dev_result = {}
    with open(args.cached_baseline, "rb") as f:
        cached_baseline_result = pickle.load(f)
    maps = []
    maps_e_table_0 = []
    maps_e_table_1 = []
    maps_table_0 = []
    maps_table_1 = []
    maps_e = []
    ndcgs = []
    ndcgs_e_table_0 = []
    ndcgs_e_table_1 = []
    ndcgs_e = []
    recall = []
    with open("./data/wikisql_entity/entity_tables.pkl", "rb") as f:
        entity_tables = pickle.load(f)
    # with open("./data/wikisql_entity/train_tables_meta.json", "r") as f:
    #     train_tables_meta = json.load(f)
    # with open("output/CER/model_v1_table_0.2_0.2_30000_1e-4_with_dist_cand_2/train_table_repr_ent_0.pickle", "rb") as f:
    #     table_repr_0 = pickle.load(f)
    # with open("output/CER/model_v1_table_0.2_0.2_30000_1e-4_with_dist_cand_2/train_table_repr_ent_-1.pickle", "rb") as f:
    #     table_repr_1 = pickle.load(f)
    entity_tables = {e:set(ts) for e, ts in entity_tables.items()}
    with open(os.path.join(args.data_dir,"dev_tables.jsonl"), 'r') as f, open(args.result_file,"w", encoding='utf-8') as f_out:
        for line in tqdm(f):
            table = json.loads(line.strip())
            table_id = table.get("_id", "")
            pgEnt = table["pgId"]
            if not pgEnt in all_entity_set:
                pgEnt = -1
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
            for i in range(num_rows):
                if entity_cells[i,0] == 1:
                    entity = rows[i][0]['surfaceLinks'][0]['target']['id']
                    entity_text = rows[i][0]['text']
                    core_entities.append([entity_text,entity])
            core_entities = [z for z in core_entities if z[1] in all_entity_set]
            if len(core_entities) < 5:
                tables_ignored += 1
                continue
            seed_entities = [z[1] for z in core_entities[:seed_num]]
            seed_entities_text = [z[0] for z in core_entities[:seed_num]]
            target_entities = set([z[1] for z in core_entities[seed_num:]])
            seeds_1, _, _, pall, pee, pce, ple, cand_e, cand_c = cached_baseline_result[table_id]
            if len(target_entities) == 0:
                tables_ignored += 1
                continue

            # assert seeds_1 == set(seed_entities)
            entity_cand = list(cand_e|cand_c)
            entity_cand = [z for z in entity_cand if z in all_entity_set]
            entity_cand.append(core_entities[0][1])
            # entity_cand_table = [entity_tables[z] for z in entity_cand]
            # cand_table = list(set().union(*entity_cand_table))
            # true_cand_table = set().union(*[entity_tables[z] for z in entity_cand if z in target_entities])
            # cand_table_repr = [[table_repr_1[z][0],table_repr_0[z][0]] for z in cand_table]
            recall.append(len(set(entity_cand)&target_entities)/len(target_entities))
            if len(set(entity_cand)&target_entities)==0:
                maps.append(0)
                maps_e_table_0.append(0)
                maps_e_table_1.append(0)
                maps_table_0.append(0)
                maps_table_1.append(0)
                maps_e.append(0)
                ndcgs.append(0)
                ndcgs_e_table_0.append(0)
                ndcgs_e_table_1.append(0)
                ndcgs_e.append(0)
                continue

            # pdb.set_trace()
            input_tok, input_tok_type, input_tok_pos, input_mask,\
                input_ent, input_ent_text, input_ent_text_length, input_ent_type, candidate_entity_set = CER_build_input(pgEnt, pgTitle, secTitle, caption, headers[0], seed_entities, seed_entities_text, entity_cand, dataset)
            # pdb.set_trace()
            input_tok = input_tok.to(args.device)
            input_tok_type = input_tok_type.to(args.device)
            input_tok_pos = input_tok_pos.to(args.device)
            input_ent = input_ent.to(args.device)
            input_ent_text = input_ent_text.to(args.device)
            input_ent_text_length = input_ent_text_length.to(args.device)
            input_ent_type = input_ent_type.to(args.device)
            input_mask = input_mask.to(args.device)
            candidate_entity_set = candidate_entity_set.to(args.device)
            # pdb.set_trace()
            with torch.no_grad():
                ent_outputs = model(input_tok, input_tok_type, input_tok_pos, input_mask,
                                input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_mask,
                                candidate_entity_set, None, None)
                ent_prediction_scores = ent_outputs[0][0].tolist()

                # pdb.set_trace()
                # table_repr = ent_outputs[1][0,1:3,:]
                # cand_table_repr = torch.FloatTensor(cand_table_repr).to(args.device)
                # cand_table_score = torch.sum(cand_table_repr*table_repr[None,:,:],dim=-1).tolist()
                # cand_table_score = F.cosine_similarity(cand_table_repr,table_repr[None,:,:],dim=-1).tolist()
                # tmp_table_score = {}
                # for i, table in enumerate(cand_table):
                #     tmp_table_score[table] = cand_table_score[i][::-1]
                p_neural = {}
                # p_neural_table_score = {}
                for i, entity in enumerate(entity_cand):
                    if entity not in cand_e:
                        p_neural[entity] = ent_prediction_scores[i]
                        # p_neural_table_score[entity] = [-1000000,-1000000]
                    else:
                        p_neural[entity] = ent_prediction_scores[i]
                        # p_neural_table_score[entity] = np.max([tmp_table_score[table] for table in entity_tables[entity]],axis=0)
                        # def softmax(z,axis=-1):
                        #     return np.exp(z)/np.sum(np.exp(z),axis=axis)
                        # p_neural_table_score[entity] = [tmp_table_score[table] for table in entity_tables[entity]]
                        # p_neural_table_score[entity] = np.sum(p_neural_table_score[entity]*softmax(p_neural_table_score[entity],0),axis=0)
            # pdb.set_trace()
            dev_result[table_id] = [target_entities, p_neural, pall, pee, pce, ple, cand_e, cand_c]
            ranked_neural = sorted(p_neural.items(),key=lambda z:z[1],reverse=True)
            ranked_neural_l = [1 if z[0] in target_entities else 0 for z in ranked_neural]
            ap_neural = average_precision(ranked_neural_l)
            maps.append(ap_neural)
            ndcg_neural = ndcg_at_k(ranked_neural_l,5)
            ndcgs.append(ndcg_neural)

            # ranked_neural_table_0 = sorted(p_neural_table_score.items(),key=lambda z:z[1][0],reverse=True)
            # ranked_neural_l_table_0 = [1 if z[0] in target_entities else 0 for z in ranked_neural_table_0 if z[0] in all_entity_set]
            # ap_neural_table_0 = average_precision(ranked_neural_l_table_0)
            # maps_e_table_0.append(ap_neural_table_0)
            # ndcg_neural_table_0 = ndcg_at_k(ranked_neural_l_table_0,5)
            # ndcgs_e_table_0.append(ndcg_neural_table_0)

            # ranked_neural_table_1 = sorted(p_neural_table_score.items(),key=lambda z:z[1][1],reverse=True)
            # ranked_neural_l_table_1 = [1 if z[0] in target_entities else 0 for z in ranked_neural_table_1 if z[0] in all_entity_set]
            # ap_neural_table_1 = average_precision(ranked_neural_l_table_1)
            # maps_e_table_1.append(ap_neural_table_1)
            # ndcg_neural_table_1 = ndcg_at_k(ranked_neural_l_table_1,5)
            # ndcgs_e_table_1.append(ndcg_neural_table_1)

            # ranked_tables_0 = sorted(tmp_table_score.items(),key=lambda z:z[1][0],reverse=True)
            # ranked_l_tables_0 = [1 if z[0] in true_cand_table else 0 for z in ranked_tables_0]
            # ap_table_0 = average_precision(ranked_l_tables_0)
            # maps_table_0.append(ap_table_0)

            # ranked_tables_1 = sorted(tmp_table_score.items(),key=lambda z:z[1][1],reverse=True)
            # ranked_l_tables_1 = [1 if z[0] in true_cand_table else 0 for z in ranked_tables_1]
            # ap_table_1 = average_precision(ranked_l_tables_1)
            # maps_table_1.append(ap_table_1)
            
            ranked_e = sorted(pee.items(),key=lambda z:z[1],reverse=True)
            ranked_e_l = [1 if z[0] in target_entities else 0 for z in ranked_e if z[0] in all_entity_set]
            ap_e = average_precision(ranked_e_l)
            maps_e.append(ap_e)
            ndcg_e = ndcg_at_k(ranked_e_l,5)
            ndcgs_e.append(ndcg_e)
            
            

            if debug and ap_neural<0.5:
                input_ent = input_ent.tolist()[0]
                pgEnt = dataset.entity_vocab[input_ent[0]]
                f_out.write('pgEnt\t%s:%d\t'%(pgEnt['wiki_title'], pgEnt['count'])+'\n')
                seed = dataset.entity_vocab[input_ent[2]]
                f_out.write('seed\t%s:%d\t'%(seed['wiki_title'], seed['count'])+'\n')
                target_entities = [dataset.entity_vocab[dataset.entity_wikid2id[z]] for z in target_entities]
                f_out.write('target:\n%s\n'%('\t'.join(['%s:%d'%(z['wiki_title'],z['count']) for z in target_entities])))
                f_out.write('\t'.join(dataset.tokenizer.convert_ids_to_tokens(input_tok.tolist()[0]))+'\n')
                f_out.write('\t'.join([str(z) for z in input_tok_type.tolist()[0]])+'\n')
                f_out.write('\t'.join([str(z) for z in sum(ent_outputs[-1]).sum(dim=1)[0,1].tolist()])+'\n')
                f_out.write('bert ap:%f\n'%ap_neural)
                for e,score in ranked_neural:
                    e = dataset.entity_vocab[dataset.entity_wikid2id[e]]
                    if e in target_entities:
                        f_out.write('[%s:%f:%d]\t'%(e['wiki_title'],score,e['count']))
                    else:
                        f_out.write('%s:%f:%d\t'%(e['wiki_title'],score,e['count']))
                f_out.write('\n')
                f_out.write('ee ap:%f\n'%ap_e)
                for e,score in ranked_e:
                    if e not in all_entity_set:
                        continue
                    e = dataset.entity_vocab[dataset.entity_wikid2id[e]]
                    if e in target_entities:
                        f_out.write('[%s:%f:%d]\t'%(e['wiki_title'],score,e['count']))
                    else:
                        f_out.write('%s:%f:%d\t'%(e['wiki_title'],score,e['count']))
                f_out.write('\n')
                # for table_id, score in ranked_tables_1[:50]:
                #     meta = train_tables_meta[table_id]
                #     f_out.write("{} | {}\t{}\t{}\n".format(meta[0],meta[1],1 if table_id in true_cand_table else 0, score[1]))
                f_out.write('\n')
                f_out.write('-'*100+'\n')

    print('tables ignored: %d'%tables_ignored)  
    print('recall: %f'%np.mean(recall))
    print("map: %f"%np.mean(maps))
    print("map EE: %f"%np.mean(maps_e))
    # print("map with table 0: %f"%np.mean(maps_e_table_0))
    # print("map with table 1: %f"%np.mean(maps_e_table_1))
    print("ndcg: %f"%np.mean(ndcgs))
    print("ndcg EE: %f"%np.mean(ndcgs_e))
    # print("ndcg with table 0: %f"%np.mean(ndcgs_e_table_0))
    # print("ndcg with table 1: %f"%np.mean(ndcgs_e_table_1))
    # print("map table with table 0: %f"%np.mean(maps_table_0))
    # print("map table with table 1: %f"%np.mean(maps_table_1))

def evaluate_CF(args, dataset, debug=True):
    def get_title(wiki_id):
        return dataset.entity_vocab[dataset.entity_wikid2id[wiki_id]]['wiki_title']
    config_class, model_class, _ = MODEL_CLASSES['CF']
    config = config_class.from_pretrained(args.config_name)
    config.output_attentions = True
    
    model = model_class(config, is_simple=True)
    checkpoint = torch.load(os.path.join(args.checkpoint, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()
    data_dir = args.data_dir
    CF = cell_filling(data_dir)
    with open(os.path.join(data_dir,"CF_dev_data.json"), 'r') as f:
        dev_data = json.load(f)
    all_recall = []
    all_precision = []
    all_precision_base = []
    all_precision_at_3 = []
    all_precision_at_3_base = []
    all_precision_at_5 = []
    all_precision_at_5_base = []
    all_precision_at_10 = []
    all_precision_at_10_base = []
    debug_file = open(os.path.join(args.checkpoint, 'CF_dev.log'), 'w', encoding='utf8')
    for table_id,pgEnt,pgTitle,secTitle,caption,(h1, h2),data_sample in tqdm(dev_data):
        tmp_table_debug_info = ''
        if debug:
            tmp_table_debug_info += '{} {} {}\n{}\t{}\n'.format(pgTitle,secTitle,caption,h1,h2)
        core_entities = []
        core_entities_text = []
        target_entities = []
        all_entity_cand = set()
        entity_cand = []
        for (core_e, core_e_text), target_e in data_sample:
            core_entities.append(core_e)
            core_entities_text.append(core_e_text)
            target_entities.append(target_e)
            cands = CF.get_cand_row(core_e, h2)
            cands = {key:value for key,value in cands.items() if key in dataset.entity_wikid2id}
            entity_cand.append(cands)
            all_entity_cand |= set(cands.keys()) 
        all_entity_cand = list(all_entity_cand)
        # pdb.set_trace()
        input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
            input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask, \
            candidate_entity_set = CF_build_input(pgEnt, pgTitle, secTitle, caption, [h1, h2], core_entities, core_entities_text, all_entity_cand, dataset)
        input_tok = input_tok.to(args.device)
        input_tok_type = input_tok_type.to(args.device)
        input_tok_pos = input_tok_pos.to(args.device)
        input_tok_mask = input_tok_mask.to(args.device)
        input_ent_text = input_ent_text.to(args.device)
        input_ent_text_length = input_ent_text_length.to(args.device)
        input_ent = input_ent.to(args.device)
        input_ent_type = input_ent_type.to(args.device)
        input_ent_mask = input_ent_mask.to(args.device)
        candidate_entity_set = candidate_entity_set.to(args.device)
        with torch.no_grad():
            _, ent_outputs = model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                            input_ent_text, input_ent_text_length, None,
                            input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
            num_sample = len(target_entities)
            ent_prediction_scores = ent_outputs[0][0,num_sample+1:].tolist()
        tp = 0
        tp_base = 0
        tp_at_3 = 0
        tp_at_3_base = 0
        tp_at_5 = 0
        tp_at_5_base = 0
        tp_at_10 = 0
        tp_at_10_base = 0
        p = 0
        # pdb.set_trace()
        for i, target_e in enumerate(target_entities):
            tmp_table_debug_info += '{}\t{}\n'.format(get_title(core_entities[i]), get_title(target_e))
            predictions = ent_prediction_scores[i]
            if len(entity_cand[i]) == 0:
                continue
            if target_e in entity_cand[i]:
                p += 1
            tmp_cand_scores = []
            for j, cand_e in enumerate(all_entity_cand):
                if cand_e in entity_cand[i]:
                    # if h2 in entity_cand[i][cand_e][0]:
                    #     tmp_cand_scores.append([cand_e, 100000])
                    # else:
                    tmp_cand_scores.append([cand_e, predictions[j]])
            sorted_cand_scores =  sorted(tmp_cand_scores, key=lambda z:z[1], reverse=True)
            sorted_cands = [z[0] for z in sorted_cand_scores]
            for cand in sorted_cands:
                if cand == target_e:
                    tmp_table_debug_info += '[{}]\t'.format(get_title(cand))
                else:
                    tmp_table_debug_info += '{}\t'.format(get_title(cand))
            tmp_table_debug_info += '\n'
            base_sorted_cands = CF.rank_cand_h(h2, entity_cand[i])
            for cand in base_sorted_cands:
                if cand == target_e:
                    tmp_table_debug_info += '[{}]\t'.format(get_title(cand))
                else:
                    tmp_table_debug_info += '{}\t'.format(get_title(cand))
            tmp_table_debug_info += '\n'
            if target_e == sorted_cands[0]:
                tp += 1
            if target_e == base_sorted_cands[0]:
                tp_base += 1
            if target_e in sorted_cands[:3]:
                tp_at_3 += 1
            if target_e in base_sorted_cands[:3]:
                tp_at_3_base += 1
            if target_e in sorted_cands[:5]:
                tp_at_5 += 1
            if target_e in base_sorted_cands[:5]:
                tp_at_5_base += 1
            if target_e in sorted_cands[:10]:
                tp_at_10 += 1
            if target_e in base_sorted_cands[:10]:
                tp_at_10_base += 1
        recall = p/len(target_entities)
        precision = tp/p if p!=0 else 0
        precision_base = tp_base/p if p!=0 else 0
        precision_at_3 = tp_at_3/p if p!=0 else 0
        precision_at_3_base = tp_at_3_base/p if p!=0 else 0
        precision_at_5 = tp_at_5/p if p!=0 else 0
        precision_at_5_base = tp_at_5_base/p if p!=0 else 0
        precision_at_10 = tp_at_10/p if p!=0 else 0
        precision_at_10_base = tp_at_10_base/p if p!=0 else 0
        all_recall.append(recall)
        if p!=0:
            all_precision.append(precision)
            all_precision_base.append(precision_base)
            all_precision_at_3.append(precision_at_3)
            all_precision_at_3_base.append(precision_at_3_base)
            all_precision_at_5.append(precision_at_5)
            all_precision_at_5_base.append(precision_at_5_base)
            all_precision_at_10.append(precision_at_10)
            all_precision_at_10_base.append(precision_at_10_base)
            if precision <= 0.8 and debug:
                tmp_table_debug_info = 'precision:{}\n'.format(precision)+tmp_table_debug_info+'*'*100+'\n'
                debug_file.write(tmp_table_debug_info)
    print("recall: %f"%np.mean(all_recall))
    print("precision: %f"%np.mean(all_precision))
    print("precision_base: %f"%np.mean(all_precision_base))
    print("precision_at_3: %f"%np.mean(all_precision_at_3))
    print("precision_at_3_base: %f"%np.mean(all_precision_at_3_base))
    print("precision_at_5: %f"%np.mean(all_precision_at_5))
    print("precision_at_5_base: %f"%np.mean(all_precision_at_5_base))
    print("precision_at_10: %f"%np.mean(all_precision_at_10))
    print("precision_at_10_base: %f"%np.mean(all_precision_at_10_base))
    debug_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--checkpoint", default=None, type=str, required=True,
                        help="The checkpoint for trained model.")
    parser.add_argument("--result_file", default=None, type=str, required=False,
                        help="The directory to store result.")
    parser.add_argument("--cached_baseline", default=None, type=str, required=False,
                        help="Baseline results")
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="Task to evaluate")
    
    ## Other parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    entity_vocab = load_entity_vocab(args.data_dir, ignore_bad_title=True, min_ent_count=2)
    entity_wikid2id = {entity_vocab[x]['wiki_id']:x for x in entity_vocab}
    dataset = WikiHybridTableDataset(args.data_dir,entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)
    

    if args.task == "CER":
        evaluate_CER(args, dataset, seed_num=2)
    elif args.task == "CF":
        evaluate_CF(args, dataset)