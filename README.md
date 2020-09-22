# TURL
This Repo contains code and data for "TURL: Table Understanding through Representation Learning". 

*We are cleaning and refactoring the code and will push the official release together with data and detailed documentation once that is done.*

## Data
Link for processed pretraining and evaluation data (https://osu.box.com/s/qqpdn4hbgjaiw2twn58aikqsxtxrwi2t). This is created based on the original WikiTables corpus (http://websail-fe.cs.northwestern.edu/TabEL/)

These are the processed raw tables we used for pre-training and creation of all test datasets, with 570171 / 5036 / 4964 tables for training/validation/testing. For row population only a small amount of tables is further filtered out which results in the final 4132 tables for testing.

In the *_tables.jsonl file, each line represents a Wikipedia table. Table content is stored in the field “tableData”, where the “target” is the actual entity links to the cell, and is also the entity to retrieve. The “id” and “title” are the Wikipedia_id and Wikipedia_title of the entity. “entityCell” and “entityColumn” shows the cells and columns that pass our filtering and are identified to contain entity information. For row population, the task is to predict the entities linked to the entity cells in the leftmost entity column.

There is also an entity_vocab.txt file contains all the entities we used in all experiments (these are the entities shown in pretraining). Each line contains vocab_id, Wikipedia_id, Wikipedia_title, freebase_mid, count of an entity.

For entity linking:
You can just use the raw split, as the linked entity to each cell is the target for entity linking. For the WikiGS corpus, please find the original release here http://www.cs.toronto.edu/~oktie/webtables/

For column type annotation:
There is a type_vocab.txt store the target types. In the *.table_col_type.json file, each example contains “table_id, pgTitle, pgEnt(Wikipedia id), secTitle, caption, headers, entities, type_annotations”

For relation extraction:
There is a relation_vocab.txt store the target relations. In the *.table_rel_extraction.json file, each example contains “table_id, pgTitle, pgId, secTitle, caption, valid_headers, entities, relations”. Note here the relation is between the subject column (leftmost) and each of the object columns (the rest). We do this to avoid checking all column pairs in the table.

All the annotation here for column type and relation extraction is get via distant supervision using linked entities and Freebase.

TODO: We will update with more instructions later, but you can try these data first.

TODO: Instruction for preparing code from original WikiTable Corpus

## Pretraining
TODO

## Evaluation
To systematically evaluate our pre-trained framework as well as facilitate research, we compile a table understanding benchmark consisting of 6 widely studied tasks covering
table interpretation (e.g., entity linking, column type annotation, relation extraction) and table augmentation (e.g., row population, cell filling, schema augmentation).

TODO: Refactoring the evaluation scripts and add instruction.

## Acknowledgement
We use the [WikiTable corpus](http://websail-fe.cs.northwestern.edu/TabEL/) for developing the dataset for pretraining and most of the evaluation. 
We also adopt the [WikiGS](http://www.cs.toronto.edu/~oktie/webtables/) for evaluation of entity linking.

We use multiple existing systems as baseline for evaluation. We took the code released by the author and made minor changes to fit our setting, please refer to the paper for more details. 


