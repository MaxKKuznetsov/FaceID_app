import os

from Model.user import User
from Model.user import DB_in_file

from mtcnn_cv2 import MTCNN

import cv2
import dlib
from skimage import io
from scipy.spatial import distance

from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

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
        # self.detector_MTCNN = MTCNN()

        ### dlib detector
        # self.dlib_shape_predictor = dlib.shape_predictor(os.path.join('Model', 'models', 'dlib_model', 'shape_predictor_68_face_landmarks.dat'))
        # self.dlib_face_recognition_model = dlib.face_recognition_model_v1(os.path.join('Model', 'models', 'dlib_model', 'dlib_face_recognition_resnet_model_v1.dat'))
        # self.dlib_detector = dlib.get_frontal_face_detector()

        ### TF_Model ###
        self.tf_model_for_embeddings = TF_Model_for_Embeddings()
        #tf_model_for_embeddings.onnx_detection()

        ### onnx model
        #self.onnx_path = os.path.join('Model', 'models', 'onnx', 'ultra_light_640.onnx')
        #self.shape_predictor_path = os.path.join('Model', 'models', 'dlib_model',
        #                                         'shape_predictor_5_face_landmarks.dat')
        #
        #self.onnx_model = onnx.load(self.onnx_path)
        #self.predictor = prepare(self.onnx_model)
        #self.ort_session = ort.InferenceSession(self.onnx_path)
        #self.input_name = self.ort_session.get_inputs()[0].name
        #
        #self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
        #self.face_aligner = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112,
        #                                                       desiredLeftEye=(0.3, 0.3))
        #################

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


class TF_Model_for_Embeddings:

    def __init__(self):
        ### onnx model
        self.onnx_path = os.path.join('Model', 'models', 'onnx', 'ultra_light_640.onnx')
        self.shape_predictor_path = os.path.join('Model', 'models', 'dlib_model',
                                                 'shape_predictor_5_face_landmarks.dat')

        self.ort_session, self.input_name, self.face_aligner = self.onnx_detection()
        #self.images_placeholder, self.embeddings, self.phase_train_placeholder, self.embedding_size, self.sess = self.tf_embeddings()

    def onnx_detection(self):

        onnx_model = onnx.load(self.onnx_path)
        predictor = prepare(onnx_model)
        ort_session = ort.InferenceSession(self.onnx_path)
        input_name = ort_session.get_inputs()[0].name

        shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
        face_aligner = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112,
                                                               desiredLeftEye=(0.3, 0.3))

        return ort_session, input_name, face_aligner

    '''
    def tf_embeddings(self):

        with tf.Graph().as_default():
            with tf.Session() as sess:

                saver_meta_path = self.onnx_path = os.path.join('DB', 'mfn_ckpt', 'mfn.ckpt.meta')
                saver_restore_path = self.onnx_path = os.path.join('DB', 'mfn_ckpt', 'mfn.ckpt')

                saver = tf.train.import_meta_graph(saver_meta_path)
                saver.restore(sess, saver_restore_path)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

        return images_placeholder, embeddings, phase_train_placeholder, embedding_size, sess
    '''
