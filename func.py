
import sqlite3
import pandas as pd

def get_database_table(database):
    cursor = database.cursor()

    # Execute the query to get the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';") # replace with appropriate query for your database

    # Fetch all the table names
    tables = cursor.fetchall()

    # Print the table names
    for table in tables:
        print(table[0])

    # Commit the changes
    database.commit()

    # Close the cursor and connection
    cursor.close()

def delete_database_table(database,name):
    # Get the cursor
    cursor = database.cursor()

    # Execute the query to delete the table
    cursor.execute(f"DROP TABLE IF EXISTS {name};") # replace "table_name" with the name of the table you want to delete

    # Commit the changes
    database.commit()

    # Close the cursor and connection
    cursor.close()

def get_shape_database(database):
    cursor = database.cursor()

    # Get the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Iterate through the tables and get their shape
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        num_rows = cursor.fetchone()[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        num_cols = len(cursor.fetchall())
        print(f"{table_name}: {num_rows} rows, {num_cols} columns")

    # Close the cursor and connection
    cursor.close()   

def read_database_tabe(conn, table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)

def save_database_tabe(df,database,table,mode="append"):
    df.to_sql(f"{table}",con=database,if_exists=mode,index=False)  

def stock_price(database,universe):
    col = read_database_tabe(database,"tickets")[universe].dropna().to_list()
    cl = read_database_tabe(database, "close")
    op = read_database_tabe(database, "open")
    high = read_database_tabe(database, "high")
    low = read_database_tabe(database, "low")
    volume = read_database_tabe(database, "volume")
    cl['TradingDate'] = pd.to_datetime(cl['TradingDate'])
    op['TradingDate'] = pd.to_datetime(op['TradingDate'])
    high['TradingDate'] = pd.to_datetime(high['TradingDate'])
    low['TradingDate'] = pd.to_datetime(low['TradingDate'])
    volume['TradingDate'] = pd.to_datetime(volume['TradingDate'])
    cl.set_index("TradingDate",inplace=True)    
    op.set_index("TradingDate",inplace=True)    
    high.set_index("TradingDate",inplace=True)    
    low.set_index("TradingDate",inplace=True)    
    volume.set_index("TradingDate",inplace=True)
    return cl[col], op[col], high[col], low[col], volume[col]