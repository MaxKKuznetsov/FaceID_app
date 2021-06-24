import cv2
import numpy as np
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QVBoxLayout, \
    QPushButton, QToolButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect

from Utility.MainWinObserver import MainWinObserver
from Utility.MainWinMeta import MainWinMeta



class SetSettings():
    '''
    screen settings
    '''
    def __init__(self):
        self.screen_settings()
        #self.saved_video_settings()

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

    def __init__(self, frame, state):
        self.frame = frame
        self.state = state
        self.frame_height, self.frame_width, self.frame_channels = self.frame.shape

        self.frame = self.draw_text(self.state, (200, 200), (255, 0, 0))

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


class VideoThread(QThread, SetSettings):
    '''
    видеопоток в формате OpenCV
    '''

    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, mController, mModel):
        super().__init__()

        #self._state = state
        #self.db_from_file = lib.DB_in_file()

        self.mModel = mModel
        self.mController = mController


    def run(self):

        # capture from web cam
        cap = cv2.VideoCapture(0)

        while True:
            ret, cv_img_in = cap.read()

            if ret:

                #print(self.mModel._state)

                screen = Srceen(cv_img_in, self.mModel._state)
                cv_img_out = screen.frame

                # emit frame to show
                self.change_pixmap_signal.emit(cv_img_out)

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        # shut down capture system
        cap.release()
        cv2.destroyAllWindows()



class View(QMainWindow, SetSettings, MainWinObserver, metaclass = MainWinMeta):
    '''
    Представления главного окна приложения
    '''

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

        #self.init_text_on_screen_app('test')

        self.modelIsChanged()

        # регистрируем представление в качестве наблюдателя
        self.mModel.addObserver(self)


    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """

        self.app_text, self.button_set = self.mModel.change_state

        #self.text_on_screen_app(self.app_text)
        self.upgrade_button()
        #self.setTestText(self.app_text)



    ### text ###
    def init_text_on_screen_app(self, text):
        # create a text label
        #self.layout = QVBoxLayout()
        #self.label = QLabel(text)
        #self.layout.addWidget(self.label)
        #self.setLayout(self.layout)

        ####
        self.image_label.setText(text)

    def upgrade_text_on_screen_app(self, text):
        self.image_label.setText(text)

    ###btn###
    def init_registration_button(self):
        btn_reg = QPushButton(self)
        #btn_reg.setText('')
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

        #video thread uding OpenCV
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

