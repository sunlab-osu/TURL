import json
import numpy as np
import pdb
import os
from tqdm import tqdm

def process_single_table(table,all_entity_set, min_num=3):
    processed_data = {}
    core_entities = {}

    table_id = table.get("_id", "")
    pgTitle = table.get("pgTitle", "").lower()
    pgEnt = table.get("pgId", -1)
    if pgEnt not in all_entity_set:
        pgEnt = -1
    secTitle = table.get("sectionTitle", "").lower()
    caption = table.get("tableCaption", "").lower()
    headers = table.get("processed_tableHeaders", [])
    rows = table.get("tableData", {})
    entity_columns = table.get("entityColumn", [])
    entity_cells = np.array(table.get("entityCell",[[]]))
    subject = table['subject_column']

    for i,j in zip(*entity_cells.nonzero()):
        if j == subject:
            e = rows[i][j]['surfaceLinks'][0]['target']['id']
            if e in all_entity_set:
                core_entities[i] = (e, rows[i][j]['text'])
    
    for i,j in zip(*entity_cells.nonzero()):
        if j!=subject and i in core_entities:
            e = rows[i][j]['surfaceLinks'][0]['target']['id']
            if e in all_entity_set:
                if j not in processed_data:
                    processed_data[j] = []
                processed_data[j].append([core_entities[i], e])
    
    final_data = []
    for j in processed_data:
        if len(processed_data[j])>min_num:
            final_data.append([table_id,pgEnt,pgTitle,secTitle,caption,[headers[0], headers[j]],processed_data[j]])
    return final_data

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
    data_dir = "./data/wikitables_v2"
    entity_vocab = load_entity_vocab(data_dir,True, min_ent_count=2)
    all_entity_set = set([item['wiki_id'] for _,item in entity_vocab.items()])
    generated_data = []
    with open(os.path.join(data_dir,"test_tables.jsonl"), 'r') as f:
        for line in tqdm(f):
            table = json.loads(line.strip())
            generated_data += process_single_table(table, all_entity_set)
    pdb.set_trace()
    print("get %d data samples in total"%len(generated_data))
    with open(os.path.join(data_dir,"CF_test_data.json"), 'w') as f:
        json.dump(generated_data, f, indent=2)
