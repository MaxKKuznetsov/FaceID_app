import os

from Model.user import User
from Model.user import DB_in_file

from mtcnn_cv2 import MTCNN

import cv2
import dlib
from skimage import io
from scipy.spatial import distance

from Utility.timer import elapsed_1arg, elapsed_2arg, elapsed_3arg

class Model:
    """
    Класс Model представляет собой реализацию модели данных.
    В модели хранятся..
    Модель предоставляет интерфейс, через который можно работать
    с хранимыми значениями.

    Модель содержит методы регистрации, удаления и оповещения
    наблюдателей.
    """

    def __init__(self):

        self.state = 'BackgroundMode'

        self.faces = []

        self.button_set = {'button_show_flag': True,
                           'button_text': 'Add new user',
                           'button_location': [],
                           'button_size': []
                           }

        #### User Data from file
        self.db_from_file = DB_in_file()
        self.known_face_encodings = self.db_from_file.known_face_encodings
        self.known_face_metadata = self.db_from_file.known_face_metadata

        ### MTCNN detector
        self.detector = MTCNN()

        ### dlib detector
        self.dlib_shape_predictor = dlib.shape_predictor(os.path.join('Model', 'dlib_model', 'shape_predictor_68_face_landmarks.dat'))
        self.dlib_face_recognition_model = dlib.face_recognition_model_v1(os.path.join('Model', 'dlib_model', 'dlib_face_recognition_resnet_model_v1.dat'))
        self.dlib_detector = dlib.get_frontal_face_detector()



        # список наблюдателей
        self._mObservers = []

    def return_button_set(self, InState):

        if (InState == 'BackgroundMode') or (InState == 'FaceIdentificationMode'):
            self.button_set = {'button_show_flag': True,
                               'button_text': 'Add new user',
                               'button_location': (),
                               'button_size': (),
                               }

        elif (InState == 'UserRegistrationMode') or (InState == 'GreetingsMode'):
            self.button_set = {'button_show_flag': True,
                               'button_text': 'EXIT',
                               'button_location': (),
                               'button_size': (),
                               }

        else:
            self.button_set = {'button_show_flag': False,
                               'button_text': '0',
                               'button_location': [],
                               'button_size': [],
                               }

    @property
    def change_state(self):
        return self.state, self.button_set

    @change_state.setter
    def change_state(self, InState):
        self.state = InState
        self.return_button_set(InState)

        self.notifyObservers()

    def addObserver(self, inObserver):
        self._mObservers.append(inObserver)

    def removeObserver(self, inObserver):
        self._mObservers.remove(inObserver)

    def notifyObservers(self):
        for x in self._mObservers:
            x.modelIsChanged()
