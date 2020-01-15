import json
import re
from tqdm import tqdm
from elastic import Elastic
import elasticsearch
import numpy as np

def build_wiki_category_index(category_file, index_name="wikipedia_category"):
    mappings = {
        "category_n": Elastic.notanalyzed_field(),
        "category_a": Elastic.analyzed_field()
    }
    elastic = Elastic(index_name)
    elastic.create_index(mappings, force=True)
    docs = {}
    count = 0
    with open(category_file, 'r') as f:
        pbar = tqdm(total=100)
        for line in f:
            wiki_id, cat = json.loads(line.strip())
            cat_a = []
            for c in cat:  # prepare analyzed version
                cat_a.append(c.replace("_", " "))
            doc = {"category_n": cat, "category_a": cat_a}
        docs[wiki_id] = doc
        if len(docs) == 10000:
            elastic.add_docs_bulk(docs)
            docs = {}
            pbar.update(10000)
    elastic.add_docs_bulk(docs)
    pbar.close()
