import sqlite3
from sqlite3 import Error
from src.utils import create_connection, create_the_table


if __name__ == '__main__':
    # Create the sqlite databse if not existing 
    create_connection(r"./french_addresses.db")

    # Query for creating the table
    create_table_query = """CREATE TABLE IF NOT EXISTS french_addresses_ban (
                            id integer PRIMARY KEY AUTOINCREMENT,
                            numero INT DEFAULT NULL,
                            nom_voie VARCHAR(255) NOT NULL,
                            code_postal INT NOT NULL,
                            nom_commune VARCHAR(255) NOT NULL,
                            x FLOAT DEFAULT NULL,
                            y FLOAT DEFAULT NULL,
                            lon FLOAT DEFAULT NULL,
                            lat FLOAT DEFAULT NULL,
                            actif TINYINT(1) DEFAULT 1
                            )"""

    # Creates the table in the sqlite database
    create_the_table(r"./french_addresses.db", create_table_query)