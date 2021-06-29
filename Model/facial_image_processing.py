import cv2
import face_recognition
import numpy as np
import pickle
import os
import math

from datetime import datetime, timedelta

#from enum import Enum
from mtcnn_cv2 import MTCNN

from Model.BD_controler import UserID_data



class FrameQualityEstimator:
    r = 0
    g = 1
    b = 2

    def __init__(self, image):
        self.image = image
        self.img_height, self.img_width = image.shape[0], image.shape[1]

        self.avg_standard_lum = 0.0  # Luminocity
        self.avg_contrast = 0.0  # Contrast

    def _get_intensity(self):
        return ((0.2126 * self.image[..., self.r]) + (0.7152 * self.image[..., self.g]) +
                (0.0722 * self.image[..., self.b])) / 255

    def estimate_brightness(self):
        """Get brightness of image in RGB format """

        intensity = self._get_intensity()
        self.avg_standard_lum = np.sum(intensity) / (self.img_height * self.img_width)
        return self.avg_standard_lum

    def estimate_contrast(self):
        """Get contrast of image in RGB format """

        if self.avg_standard_lum == 0:
            self.estimate_brightness()

        intensity = self._get_intensity()

        self.avg_contrast = math.sqrt(
            (np.sum(intensity ** 2) / (self.img_height * self.img_width)) - (self.avg_standard_lum ** 2))


    def estimate_standard_luminosity(self):
        pass

    def get_image_stats(self):
        return self.avg_standard_lum, self.avg_contrast

    def get_image_brightness(self):
        self.estimate_brightness()
        return self.avg_standard_lum

    def get_image_contrast(self):
        self.estimate_contrast()
        return self.avg_contrast

    @staticmethod
    def varianceOfLaplacian(img):
        ''''
        LAPV' algorithm (Pech2000)
        focus
        '''
        lap = cv2.Laplacian(img, ddepth=-1)  # cv2.cv.CV_64F)
        stdev = cv2.meanStdDev(lap)[1]
        s = stdev[0] ** 2
        #print(s[0])
        return s[0]

class Face:
    '''
    object face
    MTCNN:
        [{'box': [382, 186, 79, 108],
        'confidence': 0.9966862797737122,
        'keypoints': {'left_eye': (424, 228),
        'right_eye': (453, 224), 'nose': (452, 249),
        'mouth_left': (428, 272), 'mouth_right': (452, 268)}}]
    '''

    def __init__(self):
        self.box = []
        self.confidence = 0
        self.keypoints = {}
        self.face_label = ''
        self.userID = 0
        self.face_img = []
        self.face_encoding = []



class FacialImageProcessing:
    '''
    Open CV image: one frame from
    '''

    def __init__(self, frame):
        self.frame = frame
        self.faces = []
        self.face_img = []
        self.min_face_size = 5000

        user_data = UserID_data()
        self.known_face_encodings = user_data.known_face_encodings
        self.known_face_metadata = user_data.known_face_metadata

        self.faces = self.detect_face_FaceRecognition(resize_coef=2)
        #self.faces_MTCNN = self.detect_face_MTCNN()

        #there is at least one face bigger then 5000pix
        self.face_size_flag = self.face_size_test(self.faces, self.min_face_size)


    def resize_img_in(self, frame, resize_coef):

        # Resize frame to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_coef, fy=resize_coef)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        return rgb_small_frame

    def detect_face_FaceRecognition(self, resize_coef):
        '''
        face detection from face recognition, dlib, openCV
        :return:
        [{'box': [382, 186, 79, 108],
        'confidence': 0.9966862797737122,
        'keypoints': {'left_eye': (424, 228),
        'right_eye': (453, 224), 'nose': (452, 249),
        'mouth_left': (428, 272), 'mouth_right': (452, 268)}}]
        '''

        if resize_coef: #faster
            rgb_small_frame = self.resize_img_in(self.frame, 1/resize_coef)  # (120, 160, 3)
        else: #better quality
            rgb_small_frame = self.frame                      #(480, 640, 3)


        # Find all the face locations and face encodings in the current frame of video
        face_boxes = face_recognition.face_locations(rgb_small_frame)

        faces = self.face_FaceRecognition2face_MTCNN(face_boxes, resize_coef=resize_coef)

        face_label = 'FaceRec'
        faces = self.add_face_label(faces, face_label)

        return faces

    def face_FaceRecognition2face_MTCNN(self, face_boxes, resize_coef):
        '''
        :param face_boxes:
        [(223, 509, 331, 402)] - top, right, bottom, left
        --> x, y, width, height = face['box']

        where cv2.rectangle(frame, (x, y), (x + w, y + h)

        :return:
        [{'box': [382, 186, 79, 108],
        'confidence': 0.9966862797737122,
        'keypoints': {'left_eye': (424, 228),
        'right_eye': (453, 224), 'nose': (452, 249),
        'mouth_left': (428, 272), 'mouth_right': (452, 268)}}]
        '''

        faces = []
        for face_box in face_boxes:

            top, right, bottom, left = face_box

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            if resize_coef:
                top *= resize_coef
                right *= resize_coef
                bottom *= resize_coef
                left *= resize_coef

            faces.append({'box': [left, top,
                                right - left,
                                bottom - top
                                ]
                            })

            #faces.append({'box': [face_box[3], face_box[0],
            #                      face_box[1] - face_box[3],
            #                      face_box[2] - face_box[0]
            #                      ],
            #              'face_label': 'FaceRecognition'
            #              })

        return faces

    def transfer_img(self):

        height, width, channels = self.frame.shape
        if width > 640 or height > 480:
            self.frame = cv2.resize(self.frame, (min(width, 640), min(height, 480)))

        #transfer frame to from BGR to RGB format
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def detect_face_MTCNN(self):
        '''
        face detection from MTCNN
        :return:
        [{'box': [382, 186, 79, 108],
        'confidence': 0.9966862797737122,
        'keypoints': {'left_eye': (424, 228),
        'right_eye': (453, 224), 'nose': (452, 249),
        'mouth_left': (428, 272), 'mouth_right': (452, 268)}}]
        '''

        detector = MTCNN()
        faces = detector.detect_faces(self.frame)

        face_label = 'MTCNN'
        faces = self.add_face_label(faces, face_label)

        return faces

    def add_face_label(self, faces, face_label):
        for i in range(len(faces)):
            faces[i]['face_label'] = face_label

        return faces

    def crop_image(self, face):

        x, y, width, height = face['box']
        face_img = self.frame[y:y + height, x:x + width]

        return face_img

    def fase_size_calc(self, faces):
        faces_out = []

        for face in faces:
            x, y, width, height = face['box']
            face.update({'size': width * height})
            faces_out.append(face)

        return faces_out

    def face_size_test(self, faces, min_face_size):

        for face in faces:
            x, y, width, height = face['box']

            #print(width * height)

            if width * height > min_face_size:
                return True

        return False

    def face_identification(self):
        return []


