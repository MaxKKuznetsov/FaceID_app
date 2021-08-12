from PyQt5.QtSql import QSqlQuery, QSqlDatabase
import numpy as np
import pickle
import json
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
name_ind, face_image_ind, face_encoding_ind = range(3)

while query.next():
    name, face_image, face_encoding = query.value(name_ind), query.value(face_image_ind), query.value(
        face_encoding_ind)

    print(name)

    #face_image = np.fromstring(face_image)
    #face_encoding = np.fromstring(face_encoding)
    #blob = np.fromstring(blob)

    #blob = eval(blob)

    ### OK!!!

    print(len(face_image))

    face_image = np.fromstring(face_image.replace('[', '').replace(']', ''), dtype=float, sep=' ')
    #face_image = face_image.reshape(112, 112, 3)
    #face_image = face_image.resize(3, 3)

    #
    #face_image = json.loads(face_image)

    print('loading:')
    print(type(face_image))
    print(face_image.size)
    print(face_image)



con.close()
