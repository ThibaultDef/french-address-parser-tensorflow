import requests, zipfile
import gzip
from io import StringIO, BytesIO
from bs4 import BeautifulSoup as BS
import pandas as pd
from tqdm import tqdm
from src.utils import create_connection, remove_duplicate, integrate_dataframe

# Target URL we want to extract data
url = "https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/"

# Table we want to load the extracted and transformed data
table = "french_addresses_ban"

# Depending on your use case, you can also integrate house numbers of corresponding french addresses
# If you don't want to store house number in the sqlite database, the algorithm won't integrate house
# coordinates.
integrate_numero = False

# Connection to the database
connexion = create_connection(r"./french_addresses.db", return_connexion=True)

# Extraction and Load data #
response = requests.get(url)
if response.status_code == 200:
    soup = BS(response.text)
    files = soup.findAll("a", href=True)
    for file in tqdm(files):
        zip_file = file["href"]
        if ".gz" in zip_file:
            df = pd.read_csv(url+zip_file, compression="gzip", on_bad_lines='skip', delimiter=";")
            # Here you can also integrate 'numero' column if necessary 
            if integrate_numero:
                df = df[["numero", "nom_voie", "code_postal", "nom_commune", "x", "y", "lon", "lat"]]
                # Remove uncomplete data
                df = df[pd.notna(df["code_postal"])]
                df.astype({"numero": int, "nom_voie": str, "code_postal": int, "x": float, "y": float, 
                       "lon": float, "y":float})
            else:
                df = df[["nom_voie", "code_postal", "nom_commune"]]
                # Remove uncomplete data
                df = df[pd.notna(df["code_postal"])]
                df.astype({"nom_voie": str, "code_postal": int, "nom_commune": str})
                # Remove duplicates data
                df = df.drop_duplicates()
            
            df = remove_duplicate(connexion, table, df)
            integrate_dataframe(connexion, table, df)

# Close the connection to the database
connexion.close()

