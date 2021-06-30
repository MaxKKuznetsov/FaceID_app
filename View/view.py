import cv2
import numpy as np
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QVBoxLayout, \
    QPushButton, QToolButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QRect

from Utility.MainWinObserver import MainWinObserver
from Utility.MainWinMeta import MainWinMeta

from Model.facial_image_processing import FacialImageProcessing


class SetSettings:
    '''
    screen settings
    '''

    def __init__(self):
        self.screen_settings()
        # self.saved_video_settings()

    def screen_settings(self):
        self.display_width, self.display_height = 640, 480

    def saved_video_settings(self):
        self.out_video_file = os.path.join('saved_video', 'output.avi')
        self.out_video_fps = 20
        self.out_video_width, self.out_video_height = 640, 480


class Srceen:
    '''
    OpenCV frame
    '''

    def __init__(self, frame, state, faces):

        self.box_color = (0, 0, 0)
        self.frame = frame
        self.state = state
        self.faces = faces
        self.frame_height, self.frame_width, self.frame_channels = self.frame.shape

    def frame_transfer(self):

        if self.state == 'BackgroundMode':

            self.box_color = (255, 0, 0)
            self.draw_faceboxes(self.faces, self.box_color)
            self.frame = self.draw_text('BackgroundMode', (170, 30), (0, 0, 255))

        elif self.state == 'FaceIdentificationMode':
            self.box_color = (255, 0, 0)
            self.draw_faceboxes(self.faces, self.box_color)
            self.frame = self.draw_text('FaceIdentificationMode', (170, 30), (0, 0, 255))

        elif self.state == 'UserRegistrationMode':

            self.box_color = (0, 0, 255)

            self.ellipse_par()
            self.draw_ellipse()
            self.blur()

            self.draw_faceboxes(self.faces, self.box_color)
            self.frame = self.draw_text('Look at the camera', (170, 30), (0, 0, 255))

        elif self.state == 'GreetingsMode':
            # print(self.faces)
            # print(self.metadatas)

            self.box_color = (0, 255, 0)
            self.draw_text('!!!Hello!!!', (250, 70), self.box_color)

            self.draw_faceboxes_ID(self.faces, self.box_color)

        else:
            pass

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

    def ellipse_par(self):
        self.center_coordinates = (self.frame_width // 2, self.frame_height // 2)
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

    def draw_faceboxes_ID(self, faces, box_color):

        for face in faces:
            try:
                face_label = str(face['metadata']['userID'])
                self.draw_facebox_ID(face, box_color, face_label)
            except:
                self.draw_facebox(face, (255, 0, 0))


    # draw an image with detected objects
    def draw_facebox_ID(self, face, box_color, face_label):

        thickness = 2

        # get coordinates
        x, y, width, height = face['box']

        start_point = (x, y)
        end_point = (x + width, y + height)

        # face_label = face['face_label']

        # draw the box around face
        self.frame = cv2.rectangle(self.frame, start_point, end_point, box_color, thickness)

        # Draw a label with a name below the face
        cv2.rectangle(self.frame, (x - 1, y + height + 35), (x + width + 1, y + height), box_color, cv2.FILLED)
        cv2.putText(self.frame, 'ID: %s' % face_label, (x + 6, y + height + 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1)


    def draw_faceboxes(self, faces, box_color):

        for face in faces:
            self.draw_facebox(face, box_color)

    # draw an image with detected objects
    def draw_facebox(self, face, box_color):

        thickness = 2

        # get coordinates
        x, y, width, height = face['box']

        start_point = (x, y)
        end_point = (x + width, y + height)

        # draw the box around face
        self.frame = cv2.rectangle(self.frame, start_point, end_point, box_color, thickness)


class VideoThread(QThread, SetSettings):
    '''
    видеопоток в формате OpenCV
    '''

    change_pixmap_signal = pyqtSignal(np.ndarray)
    check_face_size = pyqtSignal(bool)
    metadatas_out = pyqtSignal(bool)

    def __init__(self, mController, mModel):
        super().__init__()

        self.mModel = mModel
        self.mController = mController

        self.metadatas = []

    def run(self):

        # capture from web cam
        cap = cv2.VideoCapture(0)

        while True:
            ret, cv_img_in = cap.read()

            if ret:
                state = self.mModel.state

                ### Facial Image Processing
                facal_processing = FacialImageProcessing(cv_img_in)
                # faces_MTCNN = facal_processing.detect_face_MTCNN()
                self.faces = facal_processing.faces

                ### Face identification
                if (state == 'FaceIdentificationMode') \
                        or (state == 'GreetingsMode') or (state == 'UserRegistrationMode'):
                    # face_identification
                    if self.mModel.known_face_encodings and self.mModel.known_face_metadata:
                        facal_processing.face_identification(self.mModel.known_face_encodings,
                                                             self.mModel.known_face_metadata)

                        self.faces = facal_processing.faces
                        for face in self.faces:
                            #print(face)
                            if face['metadata']:
                                self.metadatas_out.emit(True)

                if state == 'UserRegistrationMode':
                    pass

                ### Visualisation ###
                screen = Srceen(cv_img_in, state, self.faces)
                # screen.draw_faceboxes(faces_MTCNN, (255, 0, 0))
                # screen.draw_faceboxes(faces, (255, 0, 0))
                screen.frame_transfer()

                cv_img_out = screen.frame

                # emit frame to show
                self.change_pixmap_signal.emit(cv_img_out)

                # emit fase size
                self.check_face_size.emit(facal_processing.face_size_flag)

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        # shut down capture system
        cap.release()
        cv2.destroyAllWindows()


class View(QMainWindow, SetSettings, MainWinObserver, metaclass=MainWinMeta):
    '''
    Представления главного окна приложения
    '''

    face_size_test = pyqtSignal(bool)

    def __init__(self, inController, inModel, parent=None):
        """
        Конструктор принимает ссылки на модель и контроллер.
        """
        super(QMainWindow, self).__init__(parent)
        self.mController = inController
        self.mModel = inModel

        self.set_window()

        # подключаем визуальное представление
        self.initUI()

        self.btn_reg = self.init_registration_button()
        # self.face_size_check = SmartBool()

        # self.init_text_on_screen_app('test')

        self.modelIsChanged()

        # регистрируем представление в качестве наблюдателя
        self.mModel.addObserver(self)

    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """

        self.app_text, self.button_set = self.mModel.change_state

        # self.text_on_screen_app(self.app_text)
        self.upgrade_button()
        # self.setTestText(self.app_text)

    ### text ###
    def init_text_on_screen_app(self, text):
        # create a text label
        # self.layout = QVBoxLayout()
        # self.label = QLabel(text)
        # self.layout.addWidget(self.label)
        # self.setLayout(self.layout)

        ####
        self.image_label.setText(text)

    def upgrade_text_on_screen_app(self, text):
        self.image_label.setText(text)

    ###btn###
    def init_registration_button(self):
        btn_reg = QPushButton(self)
        # btn_reg.setText('')
        btn_reg.move(200, 370)
        btn_reg.resize(200, 100)

        return btn_reg

    def upgrade_button(self):
        if self.button_set['button_show_flag']:
            self.btn_reg.setText(self.button_set['button_text'])

    #########

    def set_window(self):
        '''
        set main window settings
        :return:
        '''
        self.setWindowTitle("FaceID")
        self.setGeometry(300, 300, self.display_width, self.display_height)

    def initUI(self):
        # video thread uding OpenCV
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        self.init_thread()

    def init_thread(self):
        # create the video capture thread
        self.thread = VideoThread(self.mController, self.mModel)

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""

        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)
