# TURL
This Repo contains code and data for "TURL: Table Understanding through Representation Learning". 

*We are cleaning and refactoring the code and will push the official release together with data and detailed documentation once that is done.*

## Data
Link for processed pretraining and evaluation data (https://osu.box.com/s/qqpdn4hbgjaiw2twn58aikqsxtxrwi2t). This is created based on the original WikiTables corpus (http://websail-fe.cs.northwestern.edu/TabEL/)

These are the processed raw tables we used for pre-training and creation of all test datasets, with 570171 / 5036 / 4964 tables for training/validation/testing. For row population only a small amount of tables is further filtered out which results in the final 4132 tables for testing.

In the *_tables.jsonl file, each line represents a Wikipedia table. Table content is stored in the field “tableData”, where the “target” is the actual entity links to the cell, and is also the entity to retrieve. The “id” and “title” are the Wikipedia_id and Wikipedia_title of the entity. “entityCell” and “entityColumn” shows the cells and columns that pass our filtering and are identified to contain entity information. For row population, the task is to predict the entities linked to the entity cells in the leftmost entity column.

There is also an entity_vocab.txt file contains all the entities we used in all experiments (these are the entities shown in pretraining). Each line contains vocab_id, Wikipedia_id, Wikipedia_title, freebase_mid, count of an entity.


For column type annotation:
There is a type_vocab.txt store the target types. In the *.table_col_type.json file, each example contains “table_id, pgTitle, pgEnt(Wikipedia id), secTitle, caption, headers, entities, type_annotations”

For relation extraction:
There is a relation_vocab.txt store the target relations. In the *.table_rel_extraction.json file, each example contains “table_id, pgTitle, pgId, secTitle, caption, valid_headers, entities, relations”. Note here the relation is between the subject column (leftmost) and each of the object columns (the rest). We do this to avoid checking all column pairs in the table.

All the annotation here for column type and relation extraction is get via distant supervision using linked entities and Freebase.

TODO: We will update with more instructions later, but you can try these data first.

TODO: Instruction for preparing code from original WikiTable Corpus

## Pretraining
TODO

## Finetuning & Evaluation
To systematically evaluate our pre-trained framework as well as facilitate research, we compile a table understanding benchmark consisting of 6 widely studied tasks covering
table interpretation (e.g., entity linking, column type annotation, relation extraction) and table augmentation (e.g., row population, cell filling, schema augmentation).

### Entity Linking
We use two datasets for evaluation in entity linking. One is based on our train/dev/test split, the linked entity to each cell is the target for entity linking. For the WikiGS corpus, please find the original release here http://www.cs.toronto.edu/~oktie/webtables/ .

We use entity name, together with entity description and entity type to get KB entity representation for entity linking. There are three variants for the entity linking: **0: name + description + type**, **1: name + type**, **2: name + description**.

**Evaluation**

Please see **EL** in `evaluate_task.ipynb`

**Data**

Data are stored in *[split].table_entity_linking.json*
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

### Column Type Annotation
We divide the information available in the table for column type annotation as: entity mention, table metadata and entity embedding. We experiment under 6 settings: **0: all information**, **1: only entity related**, **2: only table metadata**, **3: no entity embedding**, **4: only entity mention**, **5: only entity embedding**.

**Data**

Data are stored in *[split].table_col_type.json*
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
### Row Population
### Cell Filling
### Schema Augmentation

TODO: Refactoring the evaluation scripts and add instruction.

## Acknowledgement
We use the [WikiTable corpus](http://websail-fe.cs.northwestern.edu/TabEL/) for developing the dataset for pretraining and most of the evaluation. 
We also adopt the [WikiGS](http://www.cs.toronto.edu/~oktie/webtables/) for evaluation of entity linking.

We use multiple existing systems as baseline for evaluation. We took the code released by the author and made minor changes to fit our setting, please refer to the paper for more details. 


