import sqlite3
import pandas as pd

# 1. pulling in the fresh dataset.
print("loading new dataset...")
new_df = pd.read_csv('creditcard_2023.csv')

# 2. hooking into the local sqlite database.
conn = sqlite3.connect('fraud_detection.db')

# 3. swapping the data. 
# 'replace' wipes the old 2013 junk and drops the 2023 values into the table.
print("swapping old data for new data in the database...")
new_df.to_sql('transactions', conn, if_exists='replace', index=False)

conn.close()
print("database updated! your 'transactions' table contains the new csv values.")