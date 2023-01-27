# french-adress-parser-tensorflow

This project proposes a simple example of a french address parser by using Tensorflow framework. It is a NER model based on Transformers. Of course, many ready for use algorithms exist. The objective of this project is to propose slight modifications of existing architectures, in order to adapt existing models for a custom task.

The model architecture of this project was highly inspired by the one proposed by Tensorflow, that can be viewed via the following link:
https://keras.io/examples/nlp/ner_transformers/

We use BAN dataset for training our models that is accessible via the following link :
https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/

In the project, we give the following procedures :
1. Developpement of an ETL process to retreive all BAN dataset into a SQLite database
2. Train costum models of this project [IN COMMING]
