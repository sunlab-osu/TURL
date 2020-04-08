from data_loader.data_loaders import *
from data_loader.TR_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.hybrid_data_loaders import *
import pdb
import torch
import numpy as np
from utils.util import *

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    mode = 0
    data_dir = "./data/wikitables_v2"
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
    dev_dataset = WikiHybridTableDataset(data_dir,\
        entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=True, tokenizer = None, mode = mode)
    train_dataset = WikiHybridTableDataset(data_dir,\
        entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="train", max_length = [50, 10, 10], force_new=True, tokenizer = None, mode = mode)
    pdb.set_trace()
    dev_data_generator = HybridTableLoader(dev_dataset,10,num_workers=0,mlm_probability=0.5,ent_mlm_probability=0.5,is_train=False,use_cand=True, mode = mode)
    train_data_generator = HybridTableLoader(train_dataset,10,num_workers=0,mlm_probability=0.5,ent_mlm_probability=0.5,is_train=True,use_cand=False, mode = mode)
    for x in train_data_generator:
        pass
        pdb.set_trace()
        break
    for x in dev_data_generator:
        pass
        pdb.set_trace()
        break

    # data_dir = "data/WebQueryTable_Dataset"
    # train_dataset = WebQueryTableDataset(data_dir, src="train", force_new=True)
    # train_dataloader = TRLoader(train_dataset, 10, is_train=True)
    # for x in train_dataloader:
    #     pass
        # pdb.set_trace()
        # break

    # data_dir = "data/T2D_IO"
    # train_dataset = SemColDataset(data_dir,entity_vocab, src="train", max_cell_length=10, force_new=True)
    # train_dataloader = CTLoader(train_dataset, 10, is_train=True)
    # for x in train_dataloader:
    #     pass
    #     pdb.set_trace()
    #     break
