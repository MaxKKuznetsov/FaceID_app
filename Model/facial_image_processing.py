import cv2
import face_recognition
import numpy as np
import pickle
import os
import math

import time
from datetime import datetime

from mtcnn_cv2 import MTCNN

from keras_facenet import FaceNet

from scipy.spatial import distance

from Utility.timer import elapsed, elapsed_1arg, elapsed_2arg, elapsed_3arg


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
        # print(s[0])
        return s[0]


class FrameProcessing:
    '''
    Open CV image: one frame from
    '''

    def __init__(self, frame):
        self.frame = frame
        self.faces = []
        self.faces_MTCNN = []

        self.face_locations = []
        self.face_encodings = []
        self.face_landmarks = []

        self.face_size_flag = False

        self.resize_coef = 1

        self.confidence = 1
        self.min_face_size = 5000

    ######### dlib conveyor ##########
    def detect_face_dlib_main(self, dlib_shape_predictor, dlib_face_recognition_model, dlib_detector):
        rgb_small_frame = self.frame_transfer_in()

        # detect face
        face_locations_dlib = self.face_detection_dlib(rgb_small_frame, dlib_detector)
        face_descriptors_dlib, face_shape_dlib = self.face_encoding_dlib(rgb_small_frame, face_locations_dlib,
                                                                         dlib_shape_predictor, dlib_face_recognition_model)
        self.confidence = 1

        # create list of objects 'face'
        self.faces = self.face_dlib2face_obj(face_locations_dlib, face_descriptors_dlib, face_shape_dlib)

        # check if there is one 'big enough' face
        self.face_size_flag = self.face_size_test_all()

        return self.faces


    def face_detection_dlib(self, rgb_small_frame, dlib_detector):

        start = datetime.now()
        faces_dlib = dlib_detector(rgb_small_frame, 1)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> dlib_detector time: {elapsed}')

        return faces_dlib


    def face_encoding_dlib(self, rgb_small_frame, face_locations_dlib, dlib_shape_predictor, dlib_face_recognition_model):

        faces_descriptors, faces_shapes = [], []
        for k, d in enumerate(face_locations_dlib):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))

            start = datetime.now()
            face_shape_i = dlib_shape_predictor(rgb_small_frame, d)
            end = datetime.now()
            elapsed = (end - start).total_seconds()
            print(f'>> dlib_shape_predictor time: {elapsed}')
            ##############

            start = datetime.now()
            face_descriptor_i = dlib_face_recognition_model.compute_face_descriptor(rgb_small_frame, face_shape_i)
            end = datetime.now()
            elapsed = (end - start).total_seconds()
            print(f'>> dlib_face_recognition_model time: {elapsed}')

            faces_descriptors.append(face_descriptor_i)
            faces_shapes.append(face_shape_i)
        #############

        return faces_descriptors, faces_shapes

    def face_dlib2face_obj(self, face_locations_dlib, face_descriptors_dlib, face_shape_dlib):

        faces = []

        for i in range(len(face_locations_dlib)):

            face = Face

            face_box = face_locations_dlib[i]
            face_encoding = face_descriptors_dlib[i]
            face_shape = face_shape_dlib[i]

            left, top, right, bottom = face_box.left(), face_box.top(), face_box.right(), face_box.bottom()

            # Scale back up face locations since the frame we detected in was scaled to 1/resize_coef size
            if self.resize_coef and (self.resize_coef != 1):
                top *= self.resize_coef
                right *= self.resize_coef
                bottom *= self.resize_coef
                left *= self.resize_coef

            face.box = [left, top, right - left, bottom - top]
            face.face_encoding = face_encoding
            face.confidence = self.confidence
            face.face_landmarks = face_shape

            # calculate face size
            face.size, face.face_size_flag = face.face_size_calc(face, self.min_face_size)

            # add label to all faces
            face.face_label = 'dlib'

            faces.append(face)

        return faces

    def face_identification_dlib(self, known_face_encodings, known_face_metadata):

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        metadatas = []
        for face in self.faces:
            # Find all the face locations and face encodings in the current frame of video
            face_encoding = face.face_encoding

            # See if this face is in our list of known faces.
            metadata = self.lookup_known_face_dlib(face_encoding, known_face_encodings, known_face_metadata)
            face.metadata = metadata

    def lookup_known_face_dlib(self, face_encoding, known_face_encodings, known_face_metadata):
        """
        See if this is a face we already have in our face list
        """
        metadata = None

        # If our known face list is empty, just return nothing since we can't possibly have seen this face.
        if len(known_face_encodings) == 0:
            return metadata


        face_distances = []
        for j in range(len(known_face_encodings)):
            face_distances.append(distance.euclidean(face_encoding, known_face_encodings[j]))

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        ident_limit = 0.6
        if face_distances[best_match_index] < ident_limit:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            metadata = known_face_metadata[best_match_index]

            metadata["face_distance"] = face_distances[best_match_index]

            # print('metadata:')
            # print(metadata)

        return metadata

    #####################################



    ######### MTCNN+ conveyor ##########
    def detect_face_MTCNN_main(self, detector):

        rgb_small_frame = self.frame_transfer_in()

        # detect face
        self.faces_MTCNN = self.detect_face_MTCNN(detector, rgb_small_frame)

        #print('faces_MTCNN')
        #print(self.faces_MTCNN)

        if self.faces_MTCNN:
            self.confidence = self.faces_MTCNN[0]['confidence']

        # create list of objects 'face'
        self.faces = self.face_MTCNN2face_obj(self.faces_MTCNN)

        # check if there is one 'big enough' face
        self.face_size_flag = self.face_size_test_all()
        # check if there is one 'big enough' face
        self.face_size_flag = self.face_size_test_all()

        #encoding
        detections, embeddings = self.FaceNet_encodings(rgb_small_frame)

        print('detections')
        print(detections)

        print('embeddings')
        print(embeddings)

        # add encoding to objects 'face'
        #self.faces = self.face_MTCNN2face_obj(self.faces_MTCNN)

        return self.faces

    def FaceNet_encodings(self, frame):

        start = datetime.now()

        #embedder = FaceNet()

        # Gets a detection dict for each face
        # in an image. Each one has the bounding box and
        # face landmarks (from mtcnn.MTCNN) along with
        # the embedding from FaceNet.
        #detections = embedder.extract(frame, threshold=0.95)
        detections = []

        # If you have pre-cropped images, you can skip the
        # detection step.
        #embeddings = embedder.embeddings(frame)
        embeddings = []

        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция face_recognition.face_locations время выполнения: {elapsed}')

        return detections, embeddings



    def face_MTCNN2face_obj(self, faces_MTCNN):
        '''

        :param faces_MTCNN:
                [{'box': [382, 186, 79, 108],
        'confidence': 0.9966862797737122,
        'keypoints': {'left_eye': (424, 228),
        'right_eye': (453, 224), 'nose': (452, 249),
        'mouth_left': (428, 272), 'mouth_right': (452, 268)}}]

        :return:
        '''

        faces = []

        if not faces_MTCNN:
            return faces

        for face_MTCNN in faces_MTCNN:

            face = Face

            x, y, width, height = face_MTCNN['box']

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            if self.resize_coef and (self.resize_coef != 1):
                x *= self.resize_coef
                y *= self.resize_coef
                width *= self.resize_coef
                height *= self.resize_coef

            face.box = x, y, width, height
            face.face_encoding = []
            face.confidence = face_MTCNN['confidence']
            face.face_landmarks = face_MTCNN['keypoints']

            # calculate face size
            face.size, face.face_size_flag = face.face_size_calc(face, self.min_face_size)

            # add label to all faces
            face.face_label = 'FaceRec'

            faces.append(face)

        return faces
    ####################### FaceRecognition conveyor ############################
    def detect_face_FaceRecognition_main(self):

        rgb_small_frame = self.frame_transfer_in()

        # detect face
        self.face_locations = self.face_detection_FaceRecognition(rgb_small_frame)
        self.face_encodings, self.face_landmarks = self.face_encoding_FaceRecognition(rgb_small_frame, self.face_locations)
        self.confidence = 1

        # create list of objects 'face'
        self.faces = self.face_FaceRecognition2face_obj()

        # check if there is one 'big enough' face
        self.face_size_flag = self.face_size_test_all()

        return self.faces

    def frame_transfer_in(self):
        # Resize frame of video to 1/4 size for faster face recognition processing
        if self.resize_coef and (self.resize_coef != 1):
            small_frame = cv2.resize(self.frame, (0, 0), fx=1 / self.resize_coef, fy=1 / self.resize_coef)
        else:
            small_frame = self.frame

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        return rgb_small_frame

    def face_detection_FaceRecognition(self, rgb_small_frame):

        # Find all the face locations and face encodings in the current frame of video
        start = datetime.now()
        # self.face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        #print(f'>> функция face_recognition.face_locations время выполнения: {elapsed}')

        return face_locations

    def face_encoding_FaceRecognition(self, rgb_small_frame, face_locations):

        start = datetime.now()
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        #print(f'>> функция face_recognition.face_encodings время выполнения: {elapsed}')

        start = datetime.now()
        face_landmarks = face_recognition.api.face_landmarks(rgb_small_frame)
        elapsed = (end - start).total_seconds()
        #print(f'>> функция face_recognition.face_landmarks время выполнения: {elapsed}')
        # self.face_landmarks = []

        return face_encodings, face_landmarks

    def face_FaceRecognition2face_obj(self):
        '''

        :return:
        '''
        faces = []

        for i in range(len(self.face_locations)):

            face = Face

            face_box = self.face_locations[i]
            face_encoding = self.face_encodings[i]

            top, right, bottom, left = face_box

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            if self.resize_coef:
                top *= self.resize_coef
                right *= self.resize_coef
                bottom *= self.resize_coef
                left *= self.resize_coef

            face.box = [left, top, right - left, bottom - top]
            face.face_encoding = face_encoding
            face.confidence = self.confidence
            face.face_landmarks = self.face_landmarks

            # calculate face size
            face.size, face.face_size_flag = face.face_size_calc(face, self.min_face_size)

            # add label to all faces
            face.face_label = 'FaceRec'

            faces.append(face)

        return faces
    ##########################################################

    def transfer_img(self):

        height, width, channels = self.frame.shape
        if width > 640 or height > 480:
            self.frame = cv2.resize(self.frame, (min(width, 640), min(height, 480)))

        # transfer frame to from BGR to RGB format
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)




    def crop_image(self, face):

        x, y, width, height = face.box
        face_img = self.frame[y:y + height, x:x + width]

        return face_img

    def face_size_test_all(self):

        if not self.faces:
            return False

        size_flag_lst = []
        for face in self.faces:
            size_flag_lst.append(face.face_size_flag)

        if True in size_flag_lst:
            return True
        else:
            return False

    # @elapsed
    def face_identification_FaceRecognition(self, known_face_encodings, known_face_metadata):

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        metadatas = []
        for face in self.faces:
            # Find all the face locations and face encodings in the current frame of video
            face_encoding = face.face_encoding

            # See if this face is in our list of known faces.
            metadata = self.lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)
            face.metadata = metadata

    def lookup_known_face(self, face_encoding, known_face_encodings, known_face_metadata):
        """
        See if this is a face we already have in our face list
        """
        metadata = None

        # If our known face list is empty, just return nothing since we can't possibly have seen this face.
        if len(known_face_encodings) == 0:
            return metadata

        # Calculate the face distance between the unknown face and every face on in our known face list
        # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
        # the more similar that face was to the unknown face.
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)
        # best_match_index = np.argmax(face_distances)

        # print('best_match_index=%i' % best_match_index)

        # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
        # people will come up to the door at the same time.
        ident_limit = 0.6
        if face_distances[best_match_index] < ident_limit:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            metadata = known_face_metadata[best_match_index]

            metadata["face_distance"] = face_distances[best_match_index]

            # print('metadata:')
            # print(metadata)

        return metadata

    def frame_quality_aware(self):

        # FrameQEstimation
        FrameQEstimation = FrameQualityEstimator(self.frame)
        contrast = FrameQEstimation.get_image_contrast()
        brightness = FrameQEstimation.get_image_brightness()
        varianceOfLaplacian = FrameQEstimation.varianceOfLaplacian(self.frame)

        # FaceQEstimation
        # FaceQEstimation = FaceQNetQualityEstimator(frame)
        # FaceQuality = FaceQEstimation.estimate_quality_qnet_1frame()
        FaceQuality = 1

        # save result in faces

        if self.faces:
            # print(faces)
            for face in self.faces:
                face_confidence = round(float(face.confidence), 5)
                # print(face_confidence)

                face.contrast, face.brightness, face.focus = contrast, brightness, varianceOfLaplacian
                face.face_confidence, face.FaceQuality = face_confidence, FaceQuality

        # Qres_frame = 'c=%.2f; b=%.2f; f=%.2f' % (contrast, brightness, varianceOfLaplacian)
        # Qres_face = 'FConf=%.5f; FQual=%.5f ' % (face_confidence, FaceQuality)

        # print(Qres_frame)
        # print(Qres_face)

    def face_quality_limit(self):

        frame_height, frame_width, channels = self.frame.shape

        # print(frame_width, frame_height)

        if not self.faces:
            return False
        elif len(self.faces) > 1:
            return False
        elif not self.face_position_limit(self.faces[0], frame_height, frame_width):
            return False
        elif self.faces[0].size < 3000:
            return False
        elif self.faces[0].size > 100000:
            return False
        # elif faces[0]['focus'] < 50:
        #    return False
        # elif (faces[0]['contrast'] < 0.1) or (faces[0]['contrast'] > 0.9):
        #    return False
        # elif (faces[0]['brightness'] < 0.1) or (faces[0]['brightness'] > 0.9):
        #    return False

        elif self.faces[0].face_confidence > 0.999:
            # print('!!!!! face_confidence > 0.999 !!!!!')
            return True
        else:
            return False

    def face_position_limit(self, face, frame_height, frame_width):
        x, y, face_width, face_height = face.box

        if not (x > frame_width / 4) and ((x + face_width) < (frame_width - frame_width / 4)):
            return False
        elif not (y > frame_height / 4) and ((y + face_height) < (frame_height - frame_height / 4)):
            return False
        else:
            return True

    def new_face_image_mk(self, face):

        face2save_size = (150, 150)

        # Grab the image of the the face from the current frame of video
        x, y, face_width, face_height = face.box
        face_image = self.frame[y:y + face_height, x:x + face_width]
        face_image = cv2.resize(face_image, face2save_size)

        return face_image


class Face:
    '''
    object face
    '''

    def __init__(self):
        self.box = []
        self.face_encoding = []
        self.face_landmarks = []

        self.confidence = 0
        self.keypoints = {}
        self.face_label = ''
        # self.userID = 0

        self.size = 0
        self.face_size_flag = False

        self.contrast, self.brightness, self.focus = 0, 0, 0
        self.face_confidence, self.FaceQuality = 0, 0

        self.metadata = []
        self.face_image = []

    @staticmethod
    def face_size_calc(face, min_face_size):

        face_size_flag = False

        x, y, width, height = face.box
        size = width * height

        if size > min_face_size:
            face_size_flag = True

        return size, face_size_flag
