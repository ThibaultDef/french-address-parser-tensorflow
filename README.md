# A french address parser with Tensorflow

This project proposes a simple example of a french address parser by using Tensorflow framework. It is a NER model based on Transformers. Of course, many ready for use algorithms exist. The objective of this project is to propose slight modifications of existing architectures, in order to adapt existing models for a custom task.

The model architecture of this project was highly inspired by the one proposed by Tensorflow, that can be viewed via the following link:
https://keras.io/examples/nlp/ner_transformers/

The originality of this work is to change the word representation of input addresses (seen as sentences) by 
replacing a starting random word embedding with the one provided by Flaubert (a french BERT version). The latter 
embedding (from the encoder part of the corresponding Transformers) should help our customer model to be enougth
accurate by training it with a 'small' (in a certain sense) data sample.

We use BAN dataset for training our models that is accessible via the following link :
https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/

In the project, we give the following procedures :
1. Create the SQLite datase by executing `init_sqlite3_database.py`
2. Execute the ETL process to retreive all BAN dataset into the SQLite database, by executing `etl.py`
3. Generate training, validation and test data from the sqlite database by executing `preprocessing_data.py`
4. Train costum models of this project by executing `train.py`
5. Evaluation the custom trainned model [COMING SOON]
