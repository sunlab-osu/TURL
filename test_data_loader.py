from data_loader.data_loaders import *
import pdb
import torch
import numpy as np
from utils.util import *

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    entity_vocab = load_entity_vocab("./data/wikisql_entity", ignore_bad_title=True, min_ent_count=2)
    WikiEntityTableDataset("./data/wikisql_entity",entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=True, tokenizer = None)
    dataset = WikiEntityTableDataset("./data/wikisql_entity",entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="train", max_length = [50, 10, 10], force_new=True, tokenizer = None)
    pdb.set_trace()
    data_generator = EntityTableLoader(dataset,10,num_workers=0,mlm_probability=0.5,ent_mlm_probability=0.5)
    for x in data_generator:
        pass
        pdb.set_trace()
