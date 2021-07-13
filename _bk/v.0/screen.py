import cv2
import numpy as np
import os
import time
import textwrap

from playsound import playsound


from PyQt5 import QtWidgets

import lib



class Srceen:

    def __init__(self, frame):
        self.frame = frame
        self.frame_height, self.frame_width, self.frame_channels = self.frame.shape


    def ellipse_par(self):
        self.center_coordinates = (self.frame_width//2, self.frame_height//2)
        self.axesLength = (100, 120)
        self.angle, self.startAngle, self.endAngle = 0, 0, 360

        # Red color in BGR
        self.color = (0, 255, 0)
        # Line thickness of 5 px
        self.thickness = 1


    def draw_ellipse(self):

        cv2.ellipse(self.frame, self.center_coordinates, self.axesLength,
                    self.angle, self.startAngle, self.endAngle, self.color, self.thickness)

    def blur(self):

        blurred_img = cv2.GaussianBlur(self.frame, (21, 21), 0)

        mask = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        mask = cv2.ellipse(mask,
                           self.center_coordinates, self.axesLength, self.angle, self.startAngle, self.endAngle,
                           (255, 255, 255), -1)

        self.frame = np.where(mask == np.array([255, 255, 255]), self.frame, blurred_img)


    def draw_text(self, text, coord, color):

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = coord
        # fontScale
        fontScale = 1
        # Line thickness of 2 px
        thickness = 2

        img_out = cv2.putText(self.frame, text, org, font,
                   fontScale, color, thickness, cv2.LINE_AA)

        return img_out

    # draw an image with detected objects
    def draw_facebox1(self, face, box_color):

        face_label = face['face_label']
        # get coordinates
        x, y, width, height = face['box']

        start_point = (x, y)
        end_point = (x + width, y + height)

        thickness = 2

        # draw the box around face
        frame = cv2.rectangle(self.frame, start_point, end_point, box_color, thickness)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y + height - 35), (x + width, y + height), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, face_label, (x + 6, y + height - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame


    # draw an image with detected objects
    def draw_facebox2(self, face, Qres_frame, Qres_face):
        # get coordinates
        x, y, width, height = face['box']

        start_point = (x, y)
        end_point = (x + width, y + height)

        # Blue color in BGR
        color1 = (255, 0, 0)
        color2 = (0, 255, 0)
        color3 = (0, 0, 255)

        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # draw the box around face
        self.frame = cv2.rectangle(self.frame, start_point, end_point, color1, thickness)

        # eyes
        frame = cv2.circle(self.frame, face['keypoints']['right_eye'], 5, color2, 1)
        frame = cv2.circle(self.frame, face['keypoints']['left_eye'], 5, color2, 1)
        # nose
        frame = cv2.circle(self.frame, face['keypoints']['nose'], 5, color2, 1)
        # mouth
        frame = cv2.line(self.frame, face['keypoints']['mouth_left'], face['keypoints']['mouth_right'], color3, 3)

        # _test
        cv2.putText(self.frame, f'{Qres_frame}', (start_point[0] - 200, start_point[1]), font, 1, (0, 0, 0), 2)
        cv2.putText(self.frame, f'{Qres_face}', (start_point[0] - 200, start_point[1] + 50), font, 1, (0, 0, 0), 2)

        return self.frame


    def small_face_img_insert(self, face_img):
        pass


    def process_frame_UserRegistrationMode_step1(self):
        self.ellipse_par()
        self.draw_ellipse()
        self.blur()
        self.frame = self.draw_text('Look at the camera', (150, 30), (255, 0, 0))
        self.frame = self.draw_text('Press <<Start record>>', (150, 80), (255, 0, 0))


        return self.frame

    def process_frame_UserRegistrationMode_quality_aware_go(self, faces):

        box_color = (0, 0, 255)

        self.ellipse_par()
        self.draw_ellipse()
        self.blur()
        self.frame = self.draw_text('Look into the camera', (150, 80), (255, 0, 0))


        frame = self.drow_boxes_around_faces_quality_aware(faces, box_color)

        #frame = self.drow_boxes_around_faces(faces, box_color)
        #self.draw_facebox2(face, Qres_frame, Qres_face)

        return self.frame

    def process_frame_UserRegistrationMode_there_is_user(self, faces):

        box_color = (0, 255, 0)

        #self.frame = self.draw_text('User is registred alredy :)', (150, 200), (255, 0, 0))

        frame = self.drow_boxes_around_faces_quality_aware(faces, box_color)

        return self.frame


    def process_frame_UserRegistrationMode_step2_RecordVideo(self, dt):
        self.ellipse_par()
        self.draw_ellipse()
        self.blur()
        self.frame = self.draw_text('Recording...', (150, 30), (255, 0, 0))

        if (dt > 0) and (dt < 5):
            self.frame = self.draw_text('%i' % (5 - dt), (450, 30), (255, 0, 0))

        return self.frame

    def process_frame_UserRegistrationMode_step2_quality_aware(self, dt, face, Qres_frame, Qres_face):
        self.ellipse_par()
        self.draw_ellipse()
        self.blur()
        self.frame = self.draw_text('Recording...', (150, 30), (255, 0, 0))

        if (dt > 0) and (dt < 5):
            self.frame = self.draw_text('%i' % (5 - dt), (450, 30), (255, 0, 0))

        #self.frame = self.draw_facebox2(face, Qres_frame, Qres_face)

        return self.frame

    def process_frame_UserRegistrationMode_step2(self, dt, face, Qres_frame, Qres_face):
        self.ellipse_par()
        self.draw_ellipse()
        self.blur()
        self.frame = self.draw_text('Recording...', (150, 30), (255, 0, 0))

        if (dt > 0) and (dt < 5):
            self.frame = self.draw_text('%i' % (5 - dt), (450, 30), (255, 0, 0))

        self.frame = self.draw_facebox2(face, Qres_frame, Qres_face)

        return self.frame

    def drow_boxes_around_faces(self, faces, box_color):


        frame = self.frame

        for face in faces:
            frame = self.draw_facebox1(face, box_color)

            x, y, width, height = face['box']
            start_point = (x, y)
            end_point = (x + width, y + height)
            #frame = self.draw_text('%i' % face['size'], end_point, box_color)

        return frame

    def drow_boxes_around_faces_faceID(self, faces, box_color, userID):


        frame = self.frame

        for face in faces:
            frame = self.draw_facebox1(face, box_color)

            x, y, width, height = face['box']
            start_point = (x, y)
            end_point = (x + width, y + height)
            frame = self.draw_text('User ID: %i' % userID, (x, y + height + 30), box_color)

        return frame

    def drow_box_around_face_faceID(self, face, box_color, userID):


        frame = self.frame

        frame = self.draw_facebox1(face, box_color)

        x, y, width, height = face['box']
        frame = self.draw_text('User ID: %i' % userID, (x, y + height + 30), box_color)

        return frame

    def drow_boxes_around_faces_quality_aware(self, faces, box_color):


        frame = self.frame

        #thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        for face in faces:
            frame = self.draw_facebox1(face, box_color)

            frame_height, frame_width, channels = frame.shape

            x, y, width, height = face['box']
            start_point = (x, y)
            end_point = (x + width, y + height)

            contrast, brightness, varianceOfLaplacian = face['contrast'], face['contrast'], face['contrast']
            face_confidence, FaceQuality = face['face_confidence'], face['FaceQuality']

            #if not (x > frame_width / 10) and ((x + face_width) < (frame_width - frame_width / 10)):
            #elif not (y > frame_height / 10) and ((y + face_height) < (frame_height - frame_height / 10)):
            #box_text1 = '%i: %i; %i: %i' % (x, y, width, height)
            #print(frame_width/10,  frame_width - frame_width / 10)
            #print(frame_height/10, frame_height - frame_height / 10)

            #Qres_frame = 'c=%.2f; b=%.2f; f=%.2f' % (contrast, brightness, varianceOfLaplacian)
            #Qres_face = 'FConf=%.7f' % (face_confidence)
            #Qres_face = 'FConf=%.5f; FQual=%.5f ' % (face_confidence, FaceQuality)
            #cv2.putText(frame, f'{Qres_frame}', (start_point[0] - 200, start_point[1]), font, 1, (0, 0, 0), 2)
            #cv2.putText(frame, f'{Qres_face}', (start_point[0] - 200, start_point[1] + 50), font, 1, (0, 0, 0), 2)

            #cv2.putText(frame, box_text1, (start_point[0] - 200, start_point[1]), font, 1, (0, 0, 0), 2)


        return frame


    def process_frame_BackgroundMode(self, faces):
        box_color = (255, 0, 0)
        frame = self.draw_text('BackgroundMode', (150, 30), box_color)
        frame = self.drow_boxes_around_faces(faces, box_color)

        return frame

    def process_frame_FaceIdentificationMode(self, faces):
        box_color = (255, 0, 0)
        self.frame = self.draw_text('FaceIdentificationMode', (150, 30), box_color)
        self.frame = self.drow_boxes_around_faces(faces, box_color)

    def process_frame_GreetingsMode(self, faces, metadatas):

        box_color = (0, 255, 0)
        frame = self.draw_text('!!!Hello!!!', (250, 70), box_color)

        #print(len(metadatas))
        #print(len(faces))

        for k in range(len(metadatas)):

            try:
                metadata = metadatas[k]
                face = faces[k]
            except:
                metadata = metadatas
                face = faces


            #print(11111)
            #print(metadata)
            #print(2222)
            #print(faces[k])
            #print(333333333333)

            try:

                userID = metadata['userID']
                face_image = metadata['face_image']

                frame = self.drow_box_around_face_faceID(face, box_color, userID)

                x_position = 0
                frame[0:150 + 50 * k, x_position:x_position + 150] = face_image
                #frame = self.draw_text('User ID: %i' % userID, (0, 160), box_color)

            except:
                pass

        return frame


    def process_frame_GreetingsMode2(self, faces, metadata):

        #print(metadata)

        #userID = metadata['userID']
        #face_image = metadata['face_image']

        box_color = (0, 255, 0)
        frame = self.draw_text('You are registered already', (100, 70), box_color)
        frame = self.drow_boxes_around_faces(faces, box_color)

        #frame = self.add_face_on_frame(faces)

        return frame

    def sound_Greetings(self):
        sound_folder = 'sounds'
        photo_shoot_sound_file = 'greetings3.mp3'

        playsound(os.path.join(sound_folder, photo_shoot_sound_file))


    def sound_camera(self):
        sound_folder = 'sounds'
        photo_shoot_sound_file = 'camera_shoot.mp3'

        playsound(os.path.join(sound_folder, photo_shoot_sound_file))









