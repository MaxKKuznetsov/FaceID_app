import time
import os
import pickle
from datetime import datetime, timedelta

class SetSettings():
    def __init__(self):
        self.screen_settings()
        self.video_settings()

    def screen_settings(self):
        self.display_width, self.display_height = 640, 480

    def video_settings(self):
        self.out_video_file = os.path.join('saved_video', 'output.avi')
        #self.out_video_file = os.path.join('BestFrameSelection', 'saved_video', 'output.avi')
        self.out_video_fps = 20
        self.out_video_width, self.out_video_height = 640, 480





class Timer():
    def __init__(self):
        self.t_start = 0
        self.dt = 0

    def start_timer(self):
        self.t_start = time.time()

    def stop_timer(self):
        self.t_start = 0

    def return_time(self):
        return time.time() - self.t_start


class DB_in_file():

    def __init__(self):
        self.db_file = os.path.join('DB', 'known_faces_test.dat')
        #self.db_file = os.path.join('DB', 'known_faces111.dat')
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
