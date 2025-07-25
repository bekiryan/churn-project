import sqlite3
import pandas as pd
from ml_model.utils import load_data, split_data


def load_data_to_sql(file_path, db_path='db.sqlite'):
    """
    Load data from a CSV file into a SQLite database.
    
    Parameters:
    file_path (str): Path to the CSV file.
    db_path (str): Path to the SQLite database file. Defaults to in-memory database.
    
    Returns:
    conn (Connection): SQLite connection object.
    """

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Load data from CSV file
    data = load_data(file_path)

    # Convert the DataFrame to a SQL table
    ldata = pd.DataFrame(data[0])  # Features
    ldata['Churn'] = pd.Series(data[1])  # Labels
    
    # Write the DataFrame to a SQL table named 'data'
    ldata.to_sql('data', conn, if_exists='replace', index=False)
    
    return conn


def query_data(conn, query):
    """
    Query data from the SQLite database.
    
    Parameters:
    conn (Connection): SQLite connection object.
    query (str): SQL query string.
    
    Returns:
    DataFrame: Result of the query as a DataFrame.
    """
    return pd.read_sql_query(query, conn)

def load_data_from_sql(db_path='db.sqlite'):
    """
    Load data from SQL database into a DataFrame.
    Parameters:
    db_path (str): Path to the SQLite database file.
    Returns:
    DataFrame: Data loaded from the SQL database.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data


def main():
    # Load the dataset into SQLite
    file_path = 'dataset.csv'  # Replace with your actual file path
    conn = load_data_to_sql(file_path)
    
    # Example query to fetch all data
    query = "SELECT * FROM data"
    result = query_data(conn, query)
    
    print("Data loaded into SQLite database:")
    print(result.head())
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()