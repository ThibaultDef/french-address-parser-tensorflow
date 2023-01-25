import sqlite3
from sqlite3 import Error

def create_connection(db_file, return_connexion=False):
    """ Creates a database connection to a SQLite database.
    Args:
        - db_file (str): The name given to the .db file
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    
    finally:
        if conn and not return_connexion:
            conn.close()
        else:
            return conn


def create_the_table(db_file, create_table_query):
    """ Creates the table to store french addresses in the sqlite database.
    Args:
        - db_file (str): The sqlite databse as a .db file
        - create_table_query (str): The query to create the table
    """
    try:
        connexion = create_connection(db_file, return_connexion=True)
        connexion.execute(create_table_query)
    except Error as e:
        print(e)


if __name__ == '__main__':
    # Create the sqlite databse if not existing 
    create_connection(r"./french_addresses.db")

    # Query for creating the table
    create_table_query = """CREATE TABLE IF NOT EXISTS french_adresses (
                            id integer PRIMARY KEY AUTOINCREMENT,
                            type VARCHAR(255) DEFAULT NULL,
                            version VARCHAR(255) DEFAULT NULL,
                            housenumber INT DEFAULT NULL,
                            street VARCHAR(255) NOT NULL,
                            postcode INT NOT NULL,
                            city VARCHAR(255) NOT NULL,
                            citycode INT DEFAULT NULL,
                            x FLOAT NOT NULL,
                            y FLOAT NOT NULL,
                            importance FLOAT DEFAULT NULL,
                            actif TINYINT(1) DEFAULT 1
                            )"""

    # Creates the table in the sqlite database
    create_the_table(r"./french_addresses.db", create_table_query)