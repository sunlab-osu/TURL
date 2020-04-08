import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pickle

from transformers import BertTokenizer

import pdb
from tqdm import tqdm

RESERVED_ENT_VOCAB = {0:{'wiki_id':'[PAD]'},
                        1:{'wiki_id':'[ENT_MASK]'},
                        2:{'wiki_id':'[PG_ENT_MASK]'},
                        3:{'wiki_id':'[CORE_ENT_MASK]'}
                        }

def load_entity_vocab(data_dir):
    entity_vocab = []
    bad_title = 0
    with open(os.path.join(data_dir, 'entity_vocab.txt'), 'r', encoding="utf-8") as f:
        for line in f:
            i, entity_id, entity_title, entity_mid, count = line.strip().split('\t')
            if entity_title == '':
                print('run into entity with empty title: %s, %s'%(entity_id, entity_mid))
                bad_title += 1
            else:
                entity_vocab.append((entity_id, entity_title))
    print('total bad titles: %d'%bad_title)
    return entity_vocab

class TitleEmbeddings(nn.Module):
    """Construct the embeddings from wiki title
    """
    def __init__(self, vocab_size, hidden_size):
        super(TitleEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, title_tok):
        input_title_tok_embeds = self.word_embeddings(title_tok)
        input_title_length = torch.sum(title_tok!=0,dim=1)
        input_title_embeds = input_title_tok_embeds.sum(dim=1)/input_title_length[:, None]

        return input_title_embeds

model_dir = "../data/pre-trained_models/tiny-bert/2nd_General_TinyBERT_4L_312D"
lm_checkpoint = torch.load(model_dir+"/pytorch_model.bin")
bert_embeddings = lm_checkpoint['bert.embeddings.word_embeddings.weight']
model = TitleEmbeddings(30522, 312)
pdb.set_trace()
state_dict = model.state_dict()
state_dict['word_embeddings.weight'] = bert_embeddings
model.load_state_dict(state_dict)
tokenizer = BertTokenizer.from_pretrained(model_dir)
data_dir = '../data/wikitables_v2'
entity_vocab = load_entity_vocab(data_dir)
i = 0
batch_size = 1000
entity_embeddings = []
with tqdm(total=len(entity_vocab)) as pbar:
    while i < len(entity_vocab):
        if i + batch_size < len(entity_vocab):
            _, entity_titles = zip(*entity_vocab[i:i+batch_size])
        else:
            _, entity_titles = zip(*entity_vocab[i:])
        entity_titles_tokenized = [tokenizer.encode(' '.join(x.lower().split('_')), add_special_tokens=False) for x in entity_titles]
        max_length = max([len(x) for x in entity_titles_tokenized])
        entity_titles_padded = [x+[0]*(max_length-len(x)) for x in entity_titles_tokenized]
        entity_titles_padded = torch.LongTensor(entity_titles_padded)
        entity_embedding = model(entity_titles_padded)
        entity_embeddings.extend(entity_embedding.tolist())
        pbar.update(batch_size)
        i += batch_size
assert len(entity_embeddings) == len(entity_vocab)
entity_vocab_with_embed = {}
for i, (x, _) in enumerate(entity_vocab):
    entity_vocab_with_embed[x] = entity_embeddings[i]
entity_vocab_with_embed['[PAD]'] = bert_embeddings[tokenizer.pad_token_id].tolist()
entity_vocab_with_embed['[ENT_MASK]'] = bert_embeddings[tokenizer.mask_token_id].tolist()
entity_vocab_with_embed['[PG_ENT_MASK]'] = bert_embeddings[tokenizer.mask_token_id].tolist()
entity_vocab_with_embed['[CORE_ENT_MASK]'] = bert_embeddings[tokenizer.mask_token_id].tolist()

with open(os.path.join(data_dir, 'entity_embedding_tinybert_312.pkl'), 'wb') as f:
    pickle.dump(entity_vocab_with_embed, f)
