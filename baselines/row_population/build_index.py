import json
import re
from tqdm import tqdm
from elastic import Elastic
import elasticsearch
import numpy as np

import argparse
import os

import pdb

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
        for line in tqdm(f):
            wiki_id, cat = json.loads(line.strip())
            cat_a = []
            for c in cat:  # prepare analyzed version
                cat_a.append(c.replace("_", " "))
            doc = {"category_n": cat, "category_a": cat_a}
            docs[wiki_id] = doc
            if len(docs) == 10000:
                elastic.add_docs_bulk(docs)
                docs = {}
    elastic.add_docs_bulk(docs)


def build_wiki_table_index(table_file, index_name="table_index_wikitable_train_jan_13"):
    mappings = {
        "core_entity_n": Elastic.notanalyzed_field(),
        "all_entity_n": Elastic.notanalyzed_field(),
        "data": Elastic.analyzed_field(),
        "caption": Elastic.analyzed_field(),
        'secTitle': Elastic.analyzed_field(),
        "headings_n": Elastic.notanalyzed_field(),
        "headings": Elastic.analyzed_field(),
        "pgTitle": Elastic.analyzed_field(),
        "catchall": Elastic.analyzed_field(),
    }
    elastic = Elastic(index_name)
    elastic.create_index(mappings, force=True)
    with open(table_file, 'r') as f:
        docs = {}
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
            docs[table_id] = {
                "all_entity_n": core_entities,
                "core_entity_n": core_entities,
                "caption": caption,
                'secTitle': secTitle,
                "headings_n": headers,
                "headings": headers,
                "pgTitle": pgTitle,
                "catchall": catcallall
            }
            if len(docs) == 10000:
                elastic.add_docs_bulk(docs)
                docs = {}
        elastic.add_docs_bulk(docs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data directory.")
    args = parser.parse_args()

    category_file = os.path.join(args.data_dir, "wikipedia_categories.jsonl")
    table_file = os.path.join(args.data_dir, "train_tables.jsonl")

    build_wiki_category_index(category_file)
    build_wiki_table_index(table_file)