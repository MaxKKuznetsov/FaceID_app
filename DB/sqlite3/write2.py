#https://www.pythonforthelab.com/blog/storing-data-with-sqlite/ !!!

import sqlite3
from sqlite3 import Error

import numpy as np
import io


def adapt_array(arr):
    """
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())
    #return out.read() ?


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection):
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO users(name, face_image, face_encoding) VALUES(?, ?, ?)", ("test3", "3", "33"))
        # cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


create_users_table = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  face_image array,
  face_encoding array
);
"""

# create_user = ("INSERT INTO users(name, face_image, face_encoding) VALUES(?, ?, ?)", ("test3", "3", "33"))

connection = create_connection('test_db4dlib.sqlite')



# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

cur = connection.cursor()

#step 1: create test table
cur.execute(create_users_table)
connection.commit()

try:
    cur.execute(create_users_table)
    connection.commit()
    print("Query executed successfully")
except Error as e:
    print(f"The error '{e}' occurred")


#step 2: put array to db
#array
#x = np.arange(12).reshape(2, 6)

#put array to db
#cur.execute("insert into users (name, face_image, face_encoding) values (?, ?, ?)", ('3', x, x, ))
#connection.commit()

#step3
#read array from db
#cur.execute("select name, face_image, face_encoding from users")

#results = cur.fetchall()
#for individual_row in results:
#    print(individual_row)