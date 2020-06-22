import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pickle

from transformers import BertTokenizer

import pdb
from tqdm import tqdm

RESERVED_HEADER_VOCAB = {0:'[PAD]',
                        1:'[MASK]'}

def load_header_vocab(data_dir):
        header_vocab = []
        with open(os.path.join(data_dir, "header_vocab.txt"), "r", encoding='utf8') as f:
            for line in f:
                header = line.strip()
                header_vocab.append(header)
        return header_vocab

class HeaderEmbeddings(nn.Module):
    """Construct the embeddings from wiki title
    """
    def __init__(self, vocab_size, hidden_size):
        super(HeaderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, header_tok):
        input_header_tok_embeds = self.word_embeddings(header_tok)
        input_header_length = torch.sum(header_tok!=0,dim=1)
        input_header_embeds = input_header_tok_embeds.sum(dim=1)/input_header_length[:, None]

        return input_header_embeds

data_dir = 'data/wikitables_v2'
header_vocab = load_header_vocab(data_dir)

# model_dir = "output/hybrid/model_v1_table_0.2_0.4_0.7_30000_1e-4_with_cand_0"
# model_dir = "output/hybrid/v2/model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam"
model_dir ="data/pre-trained_models/tiny-bert/2nd_General_TinyBERT_4L_312D"
lm_checkpoint = torch.load(model_dir+"/pytorch_model.bin")
# word_embeddings = lm_checkpoint['table.embeddings.word_embeddings.weight']
word_embeddings = lm_checkpoint['bert.embeddings.word_embeddings.weight']
model = HeaderEmbeddings(30522, 312)
pdb.set_trace()
state_dict = model.state_dict()
state_dict['word_embeddings.weight'] = word_embeddings
model.load_state_dict(state_dict)
tokenizer = BertTokenizer.from_pretrained("data/pre-trained_models/tiny-bert/2nd_General_TinyBERT_4L_312D")

i = 0
batch_size = 1000
header_embeddings = []
with tqdm(total=len(header_vocab)) as pbar:
    while i < len(header_vocab):
        if i + batch_size < len(header_vocab):
            headers = header_vocab[i:i+batch_size]
        else:
            headers = header_vocab[i:]
        headers_tokenized = [tokenizer.encode(x.lower(), add_special_tokens=False) for x in headers]
        max_length = max([len(x) for x in headers_tokenized])
        headers_padded = [x+[0]*(max_length-len(x)) for x in headers_tokenized]
        headers_padded = torch.LongTensor(headers_padded)
        header_embedding = model(headers_padded)
        header_embeddings.extend(header_embedding.tolist())
        pbar.update(batch_size)
        i += batch_size
assert len(header_embeddings) == len(header_vocab)
header_vocab_with_embed = {}
for i, x in enumerate(header_vocab):
    header_vocab_with_embed[x] = header_embeddings[i]
header_vocab_with_embed['[PAD]'] = word_embeddings[tokenizer.pad_token_id].tolist()
header_vocab_with_embed['[MASK]'] = word_embeddings[tokenizer.mask_token_id].tolist()

with open(os.path.join(data_dir, 'header_embedding_312_bert.pkl'), 'wb') as f:
    pickle.dump(header_vocab_with_embed, f)
