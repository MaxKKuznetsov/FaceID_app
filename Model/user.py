import os
from datetime import datetime, timedelta
import pickle

class DB_in_file():

    def __init__(self):
        print('loading known_face_encodings')

        #self.db_file = os.path.join('DB', 'known_faces_test.dat')
        self.db_file = os.path.join('DB', 'known_faces_test1.dat')
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




