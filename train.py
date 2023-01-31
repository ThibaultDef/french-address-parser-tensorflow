import os
import numpy as np
import tensorflow as tf
from src.models import AddressParser, CustomNonPaddingTokenLoss
import pathlib

main_path = str(pathlib.Path(__file__).parent.absolute())


def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    tokens = record[: 1]
    tags = record[1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
     tokens = tf.strings.lower(tokens)
     return tokens 


# List of interested labels
labels = ["numero", "nom_voie", "code_postal", "nom_commune", "pad"]

# Take training and validation data
train_data = tf.data.TextLineDataset(r"./dataset/train_data.txt")
val_data = tf.data.TextLineDataset(r"./dataset/validation_data.txt")


# We use `padded_batch` here because each record in the dataset has a
# different length.
batch_size = 32
train_dataset = (
    train_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)
val_dataset = (
    val_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)

# Precise the pretrained model for embedding input texts
checkpoint = "flaubert/flaubert_base_cased"

# Create a callback that saves the model's weights
checkpoint_filepath = './checkpoint'
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                              save_weights_only=True,
                                              verbose=1)

# Instantiate the custom model
adress_parser_model = AddressParser(len(labels), vocab_size=None, embed_dim=None, num_heads=4, ff_dim=64, 
                                        checkpoint=checkpoint)
loss = CustomNonPaddingTokenLoss()
adress_parser_model.compile(optimizer="adam", loss=loss, run_eagerly=True)

# Train the custom model with preprocessed data come from the sqlite database
adress_parser_model.fit(train_dataset, epochs=10, callbacks=[callback])

# Summary
print(adress_parser_model.summary())
