import cv2
import face_recognition
import numpy as np
import pickle
import os
import math


from mtcnn_cv2 import MTCNN

from Utility.timer import elapsed


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

        # user_data = UserID_data()
        # self.known_face_encodings = user_data.known_face_encodings
        # self.known_face_metadata = user_data.known_face_metadata

        self.faces = self.detect_face_FaceRecognition(resize_coef=2)
        # self.faces_MTCNN = self.detect_face_MTCNN()

        # there is at least one face bigger then 5000pix
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

        if resize_coef and resize_coef != 1:  # faster
            rgb_small_frame = self.resize_img_in(self.frame, 1 / resize_coef)  # (120, 160, 3)
        else:  # better quality
            rgb_small_frame = self.frame  # (480, 640, 3)

        # Find all the face locations and face encodings in the current frame of video
        face_boxes = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_boxes)

        face_landmarks = face_recognition.api.face_landmarks(self.frame)
        #'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
        # 'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'


        confidence = 1


        faces = self.face_FaceRecognition2face_MTCNN(face_boxes, face_encodings,
                                                     resize_coef=resize_coef,
                                                     face_landmarks=face_landmarks,
                                                     confidence=confidence)

        faces = self.fase_size_calc(faces)

        face_label = 'FaceRec'
        faces = self.add_face_label(faces, face_label)

        return faces

    def face_FaceRecognition2face_MTCNN(self, face_boxes, face_encodings,
                                        resize_coef, face_landmarks, confidence):
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
        for i in range(len(face_boxes)):

            face_box = face_boxes[i]
            face_encoding = face_encodings[i]

            top, right, bottom, left = face_box

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            if resize_coef:
                top *= resize_coef
                right *= resize_coef
                bottom *= resize_coef
                left *= resize_coef

            faces.append({'box': [left, top,
                                  right - left,
                                  bottom - top],
                          'face_encoding': face_encoding,
                          'confidence': confidence,
                          'face_landmarks': face_landmarks,
                          })


        return faces

    def transfer_img(self):

        height, width, channels = self.frame.shape
        if width > 640 or height > 480:
            self.frame = cv2.resize(self.frame, (min(width, 640), min(height, 480)))

        # transfer frame to from BGR to RGB format
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

            # print(width * height)

            if width * height > min_face_size:
                return True

        return False

    #@elapsed
    def face_identification(self, known_face_encodings, known_face_metadata):

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        metadatas = []
        for k in range(len(self.faces)):

            # Find all the face locations and face encodings in the current frame of video
            face_encoding = self.faces[k]['face_encoding']

            # See if this face is in our list of known faces.
            metadata = self.lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)
            self.faces[k]['metadata'] = metadata


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



        #save result in faces

        if self.faces:
            #print(faces)
            for k in range(len(self.faces)):

                #print(self.faces[k])

                face_confidence = round(float(self.faces[k]['confidence']), 5)
                # print(face_confidence)

                self.faces[k]['contrast'], self.faces[k]['brightness'], self.faces[k]['focus'] = contrast, brightness, varianceOfLaplacian
                self.faces[k]['face_confidence'], self.faces[k]['FaceQuality'] = face_confidence, FaceQuality

        #Qres_frame = 'c=%.2f; b=%.2f; f=%.2f' % (contrast, brightness, varianceOfLaplacian)
        #Qres_face = 'FConf=%.5f; FQual=%.5f ' % (face_confidence, FaceQuality)

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
        elif self.faces[0]['size'] < 3000:
            return False
        elif self.faces[0]['size'] > 100000:
            return False
        # elif faces[0]['focus'] < 50:
        #    return False
        # elif (faces[0]['contrast'] < 0.1) or (faces[0]['contrast'] > 0.9):
        #    return False
        # elif (faces[0]['brightness'] < 0.1) or (faces[0]['brightness'] > 0.9):
        #    return False

        elif self.faces[0]['face_confidence'] > 0.999:
            #print('!!!!! face_confidence > 0.999 !!!!!')
            return True
        else:
            return False

    def face_position_limit(self, face, frame_height, frame_width):
        x, y, face_width, face_height = face['box']

        if not (x > frame_width / 4) and ((x + face_width) < (frame_width - frame_width / 4)):
            return False
        elif not (y > frame_height / 4) and ((y + face_height) < (frame_height - frame_height / 4)):
            return False
        else:
            return True

    def new_face_image_mk(self, face):

        face2save_size = (150, 150)

        # Grab the image of the the face from the current frame of video
        x, y, face_width, face_height = face['box']
        face_image = self.frame[y:y + face_height, x:x + face_width]
        face_image = cv2.resize(face_image, face2save_size)

        return face_image


