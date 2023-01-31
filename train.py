import tensorflow as tf
from src.models import AddressParser, CustomNonPaddingTokenLoss


def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    # length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[: 1]
    tags = record[1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
     tokens = tf.strings.lower(tokens)
     return tokens


# List of interested labels
labels = ["numero", "nom_voie", "code_postal", "nom_commune"]

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

adress_parser_model = AddressParser(len(labels), vocab_size=None, embed_dim=32, num_heads=4, ff_dim=64)
loss = CustomNonPaddingTokenLoss()
adress_parser_model.compile(optimizer="adam", loss=loss)
adress_parser_model.fit(train_dataset, epochs=10)