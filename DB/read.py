from PyQt5.QtSql import QSqlQuery, QSqlDatabase
import numpy as np
import pickle

import io
import sqlite3

import pypyodbc
import base64
from base64 import *

con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("test_db.sqlite")
con.open()

query = QSqlQuery()
query.exec("SELECT name, face_image, face_encoding FROM users")
name_ind, face_image_ind, face_encoding_ind, blob_ind = range(4)

while query.next():
    name, face_image, face_encoding, blob = query.value(name_ind), query.value(face_image_ind), query.value(
        face_encoding_ind), query.value(blob_ind)

    #face_image = np.fromstring(face_image)
    #face_encoding = np.fromstring(face_encoding)
    #blob = np.fromstring(blob)

    #blob = eval(blob)

    print('loading:')
    print(type(face_image))
    print(type(face_encoding))
    print(type(blob))

    print(face_image)

    print(blob)

con.close()
