# TURL
This Repo contains code and data for [Deng, Xiang, et al. "TURL: table understanding through representation learning." Proceedings of the VLDB Endowment 14.3 (2020): 307-319.](http://www.vldb.org/pvldb/vol14/p307-deng.pdf) If you use TURL in your project, please cite the following paper:

```
@article{deng2020turl,
  title={TURL: table understanding through representation learning},
  author={Deng, Xiang and Sun, Huan and Lees, Alyssa and Wu, You and Yu, Cong},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={3},
  pages={307--319},
  year={2020},
  publisher={VLDB Endowment}
}
```

![overview_0](https://drive.google.com/uc?id=13PvUgWftKf8AsbjMKIdrHydUQ7K1VKME)

* [Environment and Setup](#environment-and-setup)
* [Data](#data)
* [Pretraining](#pretraining)
* [Finetuning & Evaluation](#finetuning---evaluation)
  + [Entity Linking](#entity-linking)
  + [Column Type Annotation](#column-type-annotation)
  + [Relation Extraction](#relation-extraction)
  + [Row Population](#row-population)
  + [Cell Filling](#cell-filling)
  + [Schema Augmentation](#schema-augmentation)
* [Acknowledgement](#acknowledgement)

## Environment and Setup
The model is mainly developped using [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/). You can access the docker image we used here `docker pull xdeng/transformers:latest`

## Data
Link for processed pretraining and evaluation data, as well as the model checkpoints can be accessed [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EjZWRtslWX9CubQ92jlmNTgB74hxxXszy9BUaXG5OL5F-g?e=HN2qtD). This is created based on the original WikiTables corpus (http://websail-fe.cs.northwestern.edu/TabEL/)

## Pretraining
**Data**

The *[split]_tables.jsonl* files are used for pretraining and creation of all test datasets, with 570171 / 5036 / 4964 tables for training/validation/testing.
```
'_id': '27289759-6', # table id
'pgTitle': '2010 Santos FC season', # page title
'sectionTitle': 'Out', # section title
'tableCaption': '', # table caption
'pgId': 27289759, # wikipedia page id
'tableId': 6, # index of the table in the wikipedia page
'tableData': [[{'text': 'DF', # cell value
    'surfaceLinks': [{'surface': 'DF',
      'locType': 'MAIN_TABLE',
      'target': {'id': 649702,
       'language': 'en',
       'title': 'Defender_(association_football)'},
      'linkType': 'INTERNAL'}] # urls in the cell
      } # one for each cell,...]
      ...]
'tableHeaders': [['Pos.', 'Name', 'Moving to', 'Type', 'Source']], # row headers
'processed_tableHeaders': ['pos.', 'name', 'moving to', 'type', 'source'], # processed headers that will be used
'merged_row': [], # merged rows, we identify them by comparing the cell values
'entityCell': [[1, 1, 1, 0, 0],...], # whether the cell is an entity cell, get by checking the urls inside
'entityColumn': [0, 1, 2], # whether the column is an entity column
'column_type': [0, 0, 0, 4, 2], # more finegrained column type for debug, here we only use 0: entity columns
'unique': [0.16, 1.0, 0.75, 0, 0], # the ratio of unique entities in that column
'entity_count': 72, # total number of entities in the table
'subject_column': 1 # the column index of the subject column
```

Each line represents a Wikipedia table. Table content is stored in the field `tableData`, where the `target` is the actual entity links to the cell, and is also the entity to retrieve. The `id` and `title` are the Wikipedia_id and Wikipedia_title of the entity. `entityCell` and `entityColumn` shows the cells and columns that pass our filtering and are identified to contain entity information. 

There is also an *entity_vocab.txt* file contains all the entities we used in all experiments (these are the entities shown in pretraining). Each line contains `vocab_id, Wikipedia_id, Wikipedia_title, freebase_mid, count of an entity`.

**Get representation for a given table**
To use the pretrained model as a table encoder, use the `HybridTableMaskedLM` model class. There is a example in `evaluate_task.ipynb` for cell filling task, which also shows how to get representation for arbitrary table.


## Finetuning & Evaluation
To systematically evaluate our pre-trained framework as well as facilitate research, we compile a table understanding benchmark consisting of 6 widely studied tasks covering
table interpretation (e.g., entity linking, column type annotation, relation extraction) and table augmentation (e.g., row population, cell filling, schema augmentation).

Please see `evaluate_task.ipynb` for running evaluation for different tasks.

<p align="center"><img src="https://drive.google.com/uc?id=1Js4inKqjEYR0yLNFV9NR0g2JYGhFr4qe" width="50%" /></p>

### Entity Linking
We use two datasets for evaluation in entity linking. One is based on our train/dev/test split, the linked entity to each cell is the target for entity linking. For the WikiGS corpus, please find the original release here http://www.cs.toronto.edu/~oktie/webtables/ .

We use entity name, together with entity description and entity type to get KB entity representation for entity linking. There are three variants for the entity linking: **0: name + description + type**, **1: name + type**, **2: name + description**.

**Evaluation**

Please see **EL** in `evaluate_task.ipynb`

**Data**

Data are stored in *[split].table_entity_linking.json*, note this only contains those examples with groundtruth in the candidate set, not including empty candidate sets or all wrong examples.
```
'23235546-1', # table id
'Ivan Lendl career statistics', # page title
'Singles: 19 finals (8 titles, 11 runner-ups)', # section title
'', # caption
['outcome', 'year', ...], # headers
[[[0, 4], 'Björn Borg'], [[9, 2], 'Wimbledon'], ...], # cells, [index, entity mention (cell text)]
[['Björn Borg', 'Swedish tennis player', []], ['Björn Borg', 'Swedish swimmer', ['Swimmer']], ...], # candidate entities, this the merged set for all cells. [entity name, entity description, entity types]
[0, 12, ...] # labels, this is the index of the gold entity in the candidate entities
[[0, 1, ...], [11, 12, 13, ...], ...] # candidates for each cell
```

This will rerank the candidates with TURL. For candidate generation and full evaluation please check the notebook in `baselines/entity_linking`.

### Column Type Annotation
*We updated the CT model so now it is mutli-label classification by default and uses BCE loss which is consistent with our own dataset. If you are testing with datasets that assumes each column has single label, you can still use CrossEntropy loss. Change Line 1098 and Line 1033 in model/model.py*

We divide the information available in the table for column type annotation as: entity mention, table metadata and entity embedding. We experiment under 6 settings: **0: all information**, **1: only entity related**, **2: only table metadata**, **3: no entity embedding**, **4: only entity mention**, **5: only entity embedding**.

**Data**

Data are stored in *[split].table_col_type.json*. There is a *type_vocab.txt* store the target types.
```
'27295818-29', # table id
 '2010–11 rangers f.c. season', # page title
 27295818, # Wikipedia page id
 'overall', # section title
 '', # caption
 ['competition', 'started round', 'final position / round'], # headers
 [[[[0, 0], [26980923, 'Scottish Premier League']],
   [[1, 0], [18255941, 'UEFA Champions League']],
   ...],
  ...,
  [[[1, 2], [18255941, 'Group stage']],
   [[2, 2], [20795986, 'Round of 16']],
   ...]], # cells, [index, [entity id, entity mention (cell text)]]
 [['time.event'], ..., ['time.event']] # column type annotations, a column may have multiple types.
```
### Relation Extraction
There is a *relation_vocab.txt* store the target relations. In the *[split].table_rel_extraction.json* file, each example contains `table_id, pgTitle, pgId, secTitle, caption, valid_headers, entities, relations` similar to column type classification. Note here the relation is between the subject column (leftmost) and each of the object columns (the rest). We do this to avoid checking all column pairs in the table.
### Row Population
For row population, the task is to predict the entities linked to the entity cells in the leftmost entity column. A small amount of tables is further filtered out from *test_tables.jsonl* which results in the final 4132 tables for testing.
### Cell Filling
Please see **Pretrained and CF** in `evaluate_task.ipynb`. You can directly load the checkpoint under *pretrained*, as we do not finetune the model for cell filling.

We have three baselines for cell filling: **Exact**, **H2H**, **H2V**. The header vectors and co-occurrence statistics are pre-computed, please see *baselines/cell_filling/cell_filling.py* for details.


### Schema Augmentation
For schema augmentation, the task to populate the headers given caption and optional seed headers. We take this as a ranking problem given set of candidate headers constructed from the training data. Please see **Attribute Recommendation** in `evaluate_task.ipynb`.

## Acknowledgement
We use the [WikiTable corpus](http://websail-fe.cs.northwestern.edu/TabEL/) for developing the dataset for pretraining and most of the evaluation. 
We also adopt the [WikiGS](http://www.cs.toronto.edu/~oktie/webtables/) for evaluation of entity linking.

We use multiple existing systems as baseline for evaluation. We took the code released by the author and made minor changes to fit our setting, please refer to the paper for more details. 
