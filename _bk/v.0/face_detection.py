import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import pickle
import os
import math

from enum import Enum
from mtcnn_cv2 import MTCNN



class Thresholds(Enum):
    good_cont = 0.4
    good_lum = 0.2
    good_sharp = 2700
    good_orient = 100


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


class FacialImageProcessing:

    def __init__(self, img, known_face_encodings, known_face_metadata):
        self.detector = MTCNN()
        self.img = img
        self.faces = []
        self.face_img = []

        self.known_face_encodings, self.known_face_metadata = known_face_encodings, known_face_metadata

    def transfer_img(self):

        height, width, channels = self.img.shape
        if width > 640 or height > 480:
            self.img = cv2.resize(self.img, (min(width, 640), min(height, 480)))

        #transfer frame to from BGR to RGB format
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        return self.img

    def detect_face(self):

        faces = self.detector.detect_faces(self.img)

        return faces

    def crop_image(self, face):

        x, y, width, height = face['box']
        face_img = self.img[y:y + height, x:x + width]

        return face_img

    def fase_size_calc(self, faces):
        faces_out = []

        for face in faces:
            x, y, width, height = face['box']
            face.update({'size': width * height})
            faces_out.append(face)

        return faces_out

    def fase_size_test(self, faces):

        out = False
        for face in faces:
            x, y, width, height = face['box']

            #if width * height > 10000:
            if width * height > 3000:
                out = True

        return out

    def fase_size_test_out(self, faces):

        if not faces:
            return True

        for face in faces:
            x, y, width, height = face['box']

            if width * height < 3000:
                return True

        return False

    ###############


    def face_identification(self, frame, known_face_encodings, known_face_metadata, ident_limit):

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)



        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        metadatas = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # See if this face is in our list of known faces.
            metadata = self.lookup_known_face(face_encoding, known_face_encodings, known_face_metadata, ident_limit)
            metadatas.append(metadata)

            #return metadata

        return metadatas

    def lookup_known_face(self, face_encoding, known_face_encodings, known_face_metadata, ident_limit):
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

        #print('known_face_encodings: %i' % len(known_face_encodings))
        #print('face_encoding: %i' % len(face_encoding))
        #print('face_distances:')
        #print(face_distances)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)
        #best_match_index = np.argmax(face_distances)

        #print('best_match_index=%i' % best_match_index)

        # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
        # people will come up to the door at the same time.
        if face_distances[best_match_index] < ident_limit:

            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            metadata = known_face_metadata[best_match_index]

            metadata["face_distance"] = face_distances[best_match_index]

            #print('metadata:')
            #print(metadata)

        return metadata

    def frame_quality_aware(self, frame, face):

        face_confidence = round(float(face['confidence']), 5)
        # print(face_confidence)

        # FrameQEstimation
        FrameQEstimation = FrameQualityEstimator(frame)
        contrast = FrameQEstimation.get_image_contrast()
        brightness = FrameQEstimation.get_image_brightness()
        varianceOfLaplacian = FrameQEstimation.varianceOfLaplacian(frame)

        # FaceQEstimation
        # FaceQEstimation = FaceQNetQualityEstimator(frame)
        # FaceQuality = FaceQEstimation.estimate_quality_qnet_1frame()
        FaceQuality = 1

        Qres_frame = 'c=%.2f; b=%.2f; f=%.2f' % (contrast, brightness, varianceOfLaplacian)
        Qres_face = 'FConf=%.5f; FQual=%.5f ' % (face_confidence, FaceQuality)

        #print(Qres_frame)
        #print(Qres_face)

        return contrast, brightness, varianceOfLaplacian, face_confidence, FaceQuality

    def face_quality_limit(self, frame, faces):

        frame_height, frame_width, channels = frame.shape

        #print(frame_width, frame_height)

        if not faces:
            return False
        elif len(faces) > 1:
            return False
        elif not self.face_position_limit(faces[0], frame_height, frame_width):
            return False
        elif faces[0]['size'] < 3000:
            return False
        elif faces[0]['size'] > 100000:
            return False
        #elif faces[0]['focus'] < 50:
        #    return False
        #elif (faces[0]['contrast'] < 0.1) or (faces[0]['contrast'] > 0.9):
        #    return False
        #elif (faces[0]['brightness'] < 0.1) or (faces[0]['brightness'] > 0.9):
        #    return False


        elif faces[0]['face_confidence'] > 0.999:
            print('!!!!! face_confidence > 0.999 !!!!!')
            return True
        else:
            return False

    def face_position_limit(self, face, frame_height, frame_width):
        x, y, face_width, face_height = face['box']

        if not (x > frame_width/3) and ((x + face_width) < (frame_width - frame_width/3)):
            return False
        elif not (y > frame_height/3) and ((y + face_height) < (frame_height - frame_height/3)):
            return False
        else:
            return True

def extract_new_face(frame):

    # Find all the face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame) #!!! need to debug!!!
    face_locations = face_recognition.face_locations(frame) #!!! need to debug!!!
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face0_location = face_locations[0]
    face0_encoding = face_encodings[0]

    print(len(face0_location))
    print(len(face0_encoding))

    top, right, bottom, left = face0_location #!!! need to debug!!!
    face0_image = frame[top:bottom, left:right]
    face0_image = cv2.resize(face0_image, (150, 150))

    #!!!
    #faces = facal_processing.detect_face()
    #x, y, width, height = face['box']
    #face.update({'size': width * height})
    #faces_out.append(face)

    #facal_processing = FacialImageProcessing(frame, [], [])
    #faces = facal_processing.detect_face()
    #x, y, width, height = faces[0]['box']
    #face_image = facal_processing.crop_image(faces[0])

    ##############
    #facal_processing = FacialImageProcessing(frame, [], [])
    #faces = facal_processing.detect_face()
    #face_image = facal_processing.crop_image(faces[0])
    #x, y, width, height = faces[0]['box']
    #face_img = self.img[y:y + height, x:x + width]
    #face_locations = y, y + height, x, x + width
    #face_locations = (y, x + width, y + height, x) #(top, right, bottom, left)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)


    return face0_encoding, face0_image


















