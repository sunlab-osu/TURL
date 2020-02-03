from data_loader.data_loaders import *
import pdb
import torch
import numpy as np
from utils.util import *

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    mode = 1
    entity_vocab = load_entity_vocab("./data/wikisql_entity", ignore_bad_title=True, min_ent_count=2)
    train_dataset = WikiEntityTableDataset("./data/wikisql_entity",\
        entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="train", max_length = [50, 10, 10], force_new=True, tokenizer = None, mode = mode)
    dev_dataset = WikiEntityTableDataset("./data/wikisql_entity",\
        entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=True, tokenizer = None, mode = mode)
    pdb.set_trace()
    dev_data_generator = EntityTableLoader(dev_dataset,10,num_workers=0,mlm_probability=0.5,ent_mlm_probability=0.5,is_train=False,use_cand=True, mode = mode)
    train_data_generator = EntityTableLoader(train_dataset,10,num_workers=0,mlm_probability=0.5,ent_mlm_probability=0.5,is_train=True,use_cand=True, mode = mode)
    for x in train_data_generator:
        pass
        pdb.set_trace()
        break
    for x in dev_data_generator:
        pass
        pdb.set_trace()
        break
