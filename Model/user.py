import os
from datetime import datetime, timedelta
import pickle

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

        self.db_file = os.path.join('DB', 'test_db.sqlite')

        print('connecting with DB %s' % self.db_file)

        #self.create_empty_bd()

        db_connection_flag = self.createConnection()

        if not db_connection_flag:
            print('ERROR: db connection faled')
            sys.exit(1)

        self.load_known_faces()

    def createConnection(self):

        self.con = QSqlDatabase.addDatabase("QSQLITE")
        self.con.setDatabaseName(self.db_file)
        self.con.open()

        if not self.con.open():
            QMessageBox.critical(
                None,
                "QTableView Example - Error!",
                "Database Error: %s" % self.con.lastError().databaseText(),
            )
            return False
        return True

    def create_empty_bd(self):
        if not os.path.isfile(self.db_file):

            print('creating new db')

            # Create the connection
            con = QSqlDatabase.addDatabase("QSQLITE")
            con.setDatabaseName(self.db_file)

            # Open the connection
            if not con.open():
                print("Database Error: %s" % con.lastError().databaseText())
                sys.exit(1)

            # Create a query and execute it right away using .exec()
            createTableQuery = QSqlQuery()
            createTableQuery.exec(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
                    name VARCHAR(40) NOT NULL,
                    face_image BLOB(13000),
                    face_encoding BLOB(200)
                )
                """
            )

            con.close()

    def load_known_faces(self):

        self.known_face_encodings, self.known_face_metadata = [], []

        query = QSqlQuery()
        query.exec("SELECT name, face_image, face_encoding FROM users")
        name_ind, face_image_ind, face_encoding_ind = range(3)

        try:
            while query.next():
                name, face_image, face_encoding = query.value(name_ind), query.value(face_image_ind), query.value(face_encoding_ind)

                #face_image = pickle.loads(face_image)
                #face_encoding = pickle.loads(face_encoding)

                face_image = np.fromstring(face_image)
                face_encoding = np.fromstring(face_encoding)

                print('loading:')
                print(type(face_image))
                print(type(face_encoding))

                self.known_face_encodings.append(face_encoding)
                self.known_face_metadata.append({'userID': name, 'face_image': face_image})

                #print(name, face_image, face_encoding)

        except:
            print("No previous face data found - starting with a blank known face list.")
            self.known_face_encodings, self.known_face_metadata = [], []

        query.finish()
        print("Face ID loaded from DB")
        print('N Users=%i' % len(self.known_face_metadata))

        return self.known_face_encodings, self.known_face_metadata

    def save_known_faces(self):

        #face_data = [self.known_face_encodings, self.known_face_metadata]
        #pickle.dump(face_data, face_data_file)

        #con = QSqlDatabase.addDatabase("QSQLITE")
        #con.setDatabaseName(self.db_file)
        #con.open()
        new_face_encoding = self.known_face_encodings[-1]
        new_face_metadata = self.known_face_metadata[-1]

        new_name = new_face_metadata['userID']
        new_face_image = new_face_metadata['face_image']
        print(f'{new_name}; ' + f'{new_face_image}; ' + f'{new_face_encoding}; ')

        #new_face_image = pickle.dumps(new_face_image)
        #new_face_encoding = pickle.dumps(new_face_encoding)

        new_face_image = new_face_image.tostring()
        new_face_encoding = new_face_encoding.tostring()

        print('saving:')
        print(type(new_face_image))
        print(type(new_face_encoding))

        query = QSqlQuery(self.con)
        query.exec(
            f"""INSERT INTO users (name, face_image, face_encoding) 
            VALUES ('{new_name}', '{new_face_image}', '{new_face_encoding}')"""
        )

        print("Face ID saved to file")


        #con.close()

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


class DB_in_file():

    def __init__(self):
        print('loading known_face_encodings')

        #self.db_file = os.path.join('DB', 'known_faces.dat')
        self.db_file = os.path.join('DB', 'known_faces_test.dat')
        #self.db_file = os.path.join('DB', 'known_faces_test_none.dat')
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




