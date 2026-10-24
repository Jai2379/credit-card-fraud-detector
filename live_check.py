import sqlite3
import pandas as pd
from predict_fraud import check_transaction # reusing the brain logic

# connect to db. using the fraud_detection.db file we generated earlier.
conn = sqlite3.connect('fraud_detection.db')

# asking the database to reveal its tables. 
# this keeps the script from crashing if the table name changes later.
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_info = cursor.fetchone() # grabs the first table name found

if table_info:
    table_name = table_info[0]
    print(f"system online. connected to table: '{table_name}'")
else:
    print("error: no tables found in the database!")
    exit()

# pulling one random transaction using an f-string to inject the dynamic table name.
query = f"SELECT rowid, * FROM {table_name} ORDER BY RANDOM() LIMIT 1"
df_row = pd.read_sql(query, conn)

# extracting the unique id and the raw feature data.
record_id = df_row.iloc[0]['rowid']
real_tx = df_row.iloc[0].drop('rowid').to_dict() # stripping the id so it doesn't mess with the model input

# sending the data to mahoraga for the final verdict.
print(f"--- random live sql check ---")
print(f"analyzing transaction record id: {record_id}") 
print(check_transaction(real_tx))