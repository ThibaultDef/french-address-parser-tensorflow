import sqlite3
from sqlite3 import Error
import pandas as pd


def create_connection(db_file: str, return_connexion: bool = False):
    """ Creates a database connection to a SQLite database.
    Args:
        - db_file (str): The name given to the .db file
        - return_connexion (bool): If True, it returns the connexion.
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


def create_the_table(db_file: str, create_table_query: str):
    """ Creates the table to store french addresses in the sqlite database.
    Args:
        - db_file (str): The sqlite databse as a .db file
        - create_table_query (str): The query to create the table
    """
    try:
        connexion = create_connection(db_file, return_connexion=True)
        connexion.execute(create_table_query)
        connexion.close()
    except Error as e:
        print(e)


def export_table(connexion: sqlite3.Connection, table: str) -> pd.DataFrame:
    """ Removes duplicates from the dataframe we want to integrate data.
    Args:
        - connexion (sqlite3.Connection): Connection to the sqlite3 database
        - table (str): A table in the database

    Returns:
        - pd.DataFrame : Dataframe containing data from the selected table
    """
    return pd.read_sql(f"SELECT * FROM {table}", connexion)



def remove_duplicate(connexion: sqlite3.Connection, table: str, df: pd.DataFrame, integrate_numero: bool = False) -> pd.DataFrame:
    """ Removes duplicates from the dataframe we want to integrate data.
    Args:
        - connexion (sqlite3.Connection): Connection to the sqlite3 database
        - table (str): A table in the database
        - df (pd.DataFrame): Dataframe we want to integrate into the database whose columns should correspond
          exactly to those in the selected table
        - integrate_numero (bool): If True, we consider house number as a part of the key

    Returns:
        - pd.DataFrame : Input dataframe without data already existing in the selected table of the database
    """
    if df.shape[0] == 0:
        return df
    
    df_table = export_table(connexion, table)
    if df_table.shape[0] == 0:
        return df
    
    if integrate_numero:
        df_table = df_table[[c for c in df_table.columns.values if c not in ["id"]]]
    else:
        df = df[[c for c in df.columns.values if c not in ["id", "numero", "x", "y", "lon", "lat"]]]
        df_table = df_table[[c for c in df_table.columns.values if c not in ["id", "numero", "x", "y", "lon", "lat", "actif"]]]

    df_to_integrate = pd.concat([df, df_table], ignore_index=True)
    df_to_integrate = df_to_integrate.drop_duplicates(keep='last')

    return df_to_integrate



def integrate_dataframe(connexion: sqlite3.Connection, table: str, df: pd.DataFrame):
    """ Integrates data from a dataframe into a table of the database.
    Args:
        - connexion (sqlite3.Connection): Connection to the sqlite3 database
        - table (str): A table in the database
        - df (pd.DataFrame): Dataframe we want to integrate into the database whose columns should correspond
          exactly to those in the selected table
    """
    if df.shape[0] > 0:
        cursor = connexion.cursor()
        columns = df.columns.values
        list_data = df.to_dict('records')
        for data in list_data:
            values = [str(data[c]).replace("'", "''") for c in columns]
            values = [f"'{v}'" for v in values]
            insert_query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)})"
            cursor.execute(insert_query)
            connexion.commit()