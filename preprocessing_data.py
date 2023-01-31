import random
from sklearn.model_selection import train_test_split
from src.utils import create_connection, export_table


# Table we want to load the extracted and transformed data
table = "french_addresses_ban"

# List of interested columns/labels
columns = ["numero", "nom_voie", "code_postal", "nom_commune"]

# Dictionnary assiocating labels to tags
dict_label_tags = {c: i+1 for i, c in enumerate(columns)}
dict_label_tags["pad"] = 0

# Connection to the SQLite database
connexion = create_connection(r"./french_addresses.db", return_connexion=True)

# Export data from the database
df_data = export_table(connexion, table, shuffle=True, limit_row=1000)

# The house numbers have not been stored in the database, because it would be too heavy. In that case,
# we randomly give a house number to each row.
df_data["numero"] = [random.randint(0, 100) for i in range(df_data.shape[0])]

# Concatenation to create full addresses per row
df_data["full_address"] = df_data[columns].apply(lambda x: ' '.join([str(t) for t in x]), axis=1)

# Add labels for each tokens
df_data["labels"] = df_data[columns].apply(
    lambda x: '\t'.join(['\t'.join([str(dict_label_tags[columns[i]])]*len(str(t).split())) for i, t in enumerate(x)]), axis=1)

# Contenation of full addresses and labels
df_data["full_address"] = df_data["full_address"] + "\t" + df_data["labels"] + "\n"

# Split data into train, validation and test data
df_train_data, df_test_data = train_test_split(df_data, test_size=0.40, random_state=42)
df_validation_data, df_test_data = train_test_split(df_test_data, test_size=0.20, random_state=42)

# Create .txt files for feeding it to a neural network
with open(r'./dataset/train_data.txt', 'w') as ftr:
    ftr.writelines(df_train_data['full_address'].tolist())

with open(r'./dataset/validation_data.txt', 'w') as fv:
    fv.writelines(df_validation_data['full_address'].tolist())

with open(r'./dataset/test_data.txt', 'w') as fte:
    fte.writelines(df_test_data['full_address'].tolist())

