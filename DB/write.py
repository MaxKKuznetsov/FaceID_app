from PyQt5.QtSql import QSqlQuery, QSqlDatabase
import numpy as np
import pickle

import io
import sqlite3

import pypyodbc
import base64
from base64 import *

def adapt_array(arr):
    return arr.astype('float32').tobytes()


def convert_array(text):
    return np.frombuffer(text, dtype='float32')

def adapt_array(arr):
    """
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


array_in1 = np.array([[1, 2, 3], [4, 5, 6]])
array_in2 = np.array([[11, 21, 31], [41, 51, 61]])

con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("test_db.sqlite")
con.open()

# Creating a query for later execution using .prepare()
insertDataQuery = QSqlQuery()
insertDataQuery.prepare(
    """
    INSERT INTO users (
        name,
        face_image,
        face_encoding,
        blob
    )
    VALUES (?, ?, ?, ?)
    """
)

#
#array1 = adapt_array(array1)  # no (
#array1 = array1.tostring() # no (
#array1 = base64.encodebytes(array1) #no
#array1 = array1.tostring() #no
#array1 = tuple(array_in1)
#array1 = array_in1.tobytes()

#convert to list, then to string, then store as VARCHAR(20000) in mysql
array1 = str(array_in1.tolist()) #ok
#Note: to retreive from mysql,you can convert to array using eval:
#myArray = eval(mycolumn)

# Converts np.array to TEXT when inserting
#sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
#sqlite3.register_converter("array", convert_array)

print(type(array1))
print(array1)

# Sample data
data = [
    ("name1", array1, 111, array1)
]

# Use .addBindValue() to insert data
for name, face_image, face_encoding, blob in data:
    insertDataQuery.addBindValue(name)
    insertDataQuery.addBindValue(face_image)
    insertDataQuery.addBindValue(face_encoding)
    insertDataQuery.addBindValue(blob)
    insertDataQuery.exec()

con.commit()
con.close()