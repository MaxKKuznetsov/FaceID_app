import os
from datetime import datetime, timedelta
import pickle
import io

import sqlite3
from sqlite3 import Error

import numpy as np

from Utility.timer import elapsed_1arg, elapsed_2arg, elapsed_3arg

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTableView,
)


class DB_Qt_QSQLITE():
    def __init__(self):

        self.db_file = os.path.join('DB', 'sqlite3', 'test_db.sqlite')

        self.shape_face_image = (112, 112, 3)
        self.shape_face_encoding = (192,)

        print('connecting with DB %s' % self.db_file)


        # create table
        # self.create_empty_table()

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)

        self.load_known_faces()


    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
            print("Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

        return connection

    def create_empty_table(self):

        create_users_table_query = """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          face_image array,
          face_encoding array
        );
        """

        self.execute_query(create_users_table_query)

    def execute_query(self, query):

        cursor = self.connection.cursor()

        try:
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")
            out = None

        return out

    def adapt_array(self, arr):
        """
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
        # return out.read() ?

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def load_known_faces(self):

        connection = self.create_connection(self.db_file)

        cur = connection.cursor()

        self.known_face_encodings, self.known_face_metadata = [], []

        try:
            cur.execute("select name, face_image, face_encoding from users")
            results = cur.fetchall()

            for individual_row in results:
                name = individual_row[0]
                face_image = individual_row[1]
                face_encoding = individual_row[2]

                self.known_face_encodings.append(face_encoding)
                self.known_face_metadata.append({'userID': name, 'face_image': face_image})

        except Error as e:
            print(f"The error '{e}' occurred")
            self.known_face_encodings, self.known_face_metadata = [], []

        cur.close()
        connection.close()

        print("Face ID loaded from DB")
        print('N Users=%i' % len(self.known_face_metadata))

        return self.known_face_encodings, self.known_face_metadata

    def save_known_faces(self):

        connection = self.create_connection(self.db_file)

        cur = connection.cursor()

        new_face_encoding = self.known_face_encodings[-1]
        new_face_metadata = self.known_face_metadata[-1]

        new_name = new_face_metadata['userID']
        new_face_image = new_face_metadata['face_image']

        # saving to db
        try:
            cur.execute("insert into users (name, face_image, face_encoding) values (?, ?, ?)",
                        (str(new_name), new_face_image, new_face_encoding,))

            connection.commit()

        except Error as e:
            print(f"The error '{e}' occurred")

        cur.close()
        connection.close()

        print('saving:')
        print(type(new_face_image))
        print(len(new_face_image))
        print(type(new_face_encoding))
        print(len(new_face_encoding))

    def register_new_face(self, new_face_encoding, new_face_image, userID):
        """
        Add a new person to our list of known faces
        """
        # Add the face encoding to the list of known faces
        self.known_face_encodings.append(new_face_encoding)

        # Add a matching dictionary entry to our metadata list.
        # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
        self.known_face_metadata.append({
            "userID": userID,
            "first_seen": datetime.now(),
            "first_seen_this_interaction": datetime.now(),
            "last_seen": datetime.now(),
            "seen_count": 1,
            "seen_frames": 1,
            "face_image": new_face_image,
            "face_distance": 0,
        })

    def __del__(self):
        self.connection.close()


class DB_in_file():

    def __init__(self):
        print('loading known_face_encodings')

        # self.db_file = os.path.join('DB', 'known_faces.dat')
        self.db_file = os.path.join('DB', 'known_faces_test.dat')
        # self.db_file = os.path.join('DB', 'known_faces_test_none.dat')
        self.create_empty_file()
        self.load_known_faces()

    def create_empty_file(self):
        if not os.path.isfile(self.db_file):
            with open(self.db_file, 'wb') as file:
                pickle.dump(dict, file)
            file.close()

    def load_known_faces(self):

        try:
            with open(self.db_file, 'rb') as face_data_file:
                self.known_face_encodings, self.known_face_metadata = pickle.load(face_data_file)
                print("Face ID loaded from file")
                print('N Users=%i' % len(self.known_face_metadata))

        except:
            print("No previous face data found - starting with a blank known face list.")
            self.known_face_encodings, self.known_face_metadata = [], []

        return self.known_face_encodings, self.known_face_metadata

    def save_known_faces(self):
        with open(self.db_file, "wb") as face_data_file:
            face_data = [self.known_face_encodings, self.known_face_metadata]
            pickle.dump(face_data, face_data_file)
            print("Face ID saved to file")

    def register_new_face(self, new_face_encoding, new_face_image, userID):
        """
        Add a new person to our list of known faces
        """
        # Add the face encoding to the list of known faces
        self.known_face_encodings.append(new_face_encoding)

        # Add a matching dictionary entry to our metadata list.
        # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
        self.known_face_metadata.append({
            "userID": userID,
            "first_seen": datetime.now(),
            "first_seen_this_interaction": datetime.now(),
            "last_seen": datetime.now(),
            "seen_count": 1,
            "seen_frames": 1,
            "face_image": new_face_image,
            "face_distance": 0,
        })


class User:
    '''
    User ID data drom file
    '''

    def __init__(self):
        pass
