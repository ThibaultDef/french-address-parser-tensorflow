import tensorflow as tf
from src.utils import map_record_to_training_data, lowercase_and_convert_to_ids
from src.models import AddressParser, CustomNonPaddingTokenLoss

# List of interested columns/labels
columns = ["numero", "nom_voie", "code_postal", "nom_commune"]

# Dictionnary assiocating labels to tags
dict_label_tags = {c: i+1 for i, c in enumerate(columns)}
dict_label_tags["pad"] = 0

# Instantiate the custom model
model = AddressParser(len(dict_label_tags), num_heads=4, ff_dim=64)

# Load the trained model
model.load_weights('./model')
loss = CustomNonPaddingTokenLoss()
model.compile(optimizer="adam", loss=loss, run_eagerly=True, metrics=['accuracy'])

# Take test data
test_data = tf.data.TextLineDataset(r"./dataset/test_data.txt")
test_dataset = (
    test_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
)
                                                    
# Evaluation the model
loss, accuracy = model.evaluate(test_dataset)
print("loss :", loss)
print("accuracy :", accuracy)
                                                    

