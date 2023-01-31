import os
import numpy as np
import tensorflow as tf
from src.utils import map_record_to_training_data, lowercase_and_convert_to_ids
from src.models import AddressParser, CustomNonPaddingTokenLoss
import pathlib
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Arguments for the training step')
parser.add_argument('--epochs', type=int, help='Number of epochs for the training step')
args = parser.parse_args()

# Retrieve arguments
epochs = args.epochs if isinstance(args.epochs, int) else 50
print(epochs)

# Path from root to the project folder
main_path = str(pathlib.Path(__file__).parent.absolute())

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
checkpoint_filepath = './model'
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                              save_weights_only=True,
                                              verbose=1)

# Instantiate the custom model
adress_parser_model = AddressParser(len(labels), num_heads=4, ff_dim=64, checkpoint=checkpoint)
loss = CustomNonPaddingTokenLoss(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
adress_parser_model.compile(optimizer=optimizer, loss=loss, run_eagerly=True, metrics=['accuracy'])

# Train the custom model with preprocessed data come from the sqlite database
adress_parser_model.fit(train_dataset, epochs=epochs, callbacks=[callback])

# Summary
print(adress_parser_model.summary())
