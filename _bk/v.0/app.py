#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys


from datetime import datetime
import time


from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QVBoxLayout, \
    QPushButton, QToolButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject



### my lib ###
import lib
from state_pattern_classes import State, BackgroundMode, \
    FaceIdentificationMode, UserRegistrationMode1, UserRegistrationMode2, GreetingsMode, \
    UserRegistrationMode1_quality_aware

import face_detection

from screen import Srceen



################
class VideoThread(QThread, lib.SetSettings):
    '''
    open cv image
    '''

    change_pixmap_signal = pyqtSignal(np.ndarray)
    emit_start_best_frame_selection_flag = pyqtSignal(bool)
    emit_state = pyqtSignal(str)

    def __init__(self, state: State):
        super().__init__()

        self._state = state

        self.db_from_file = lib.DB_in_file()


        ########################
        self.record_video_flag = False
        self.video_is_recorded_flag = False
        self.runing_best_frame_sellection = False

        self.DB_connection_flag = True

        self.timer = lib.Timer()
        self.dt = 0

        self._run_flag = True

        self.load_DB_flag = True
        self.save_in_DB_flag = False

        self.identified_metadata = []
        #############################

    def record_video(self, frame):

        if self.timer.return_time() < 5:
            self.video_writer.write(frame)

        else:
            self.record_video_flag = False
            self.timer.stop_timer()
            self.video_writer.release()
            self.video_is_recorded_flag = True
            print('stop recording')

    def first_frame_in_series(self):
        print('start recording')
        self.timer.start_timer()

        try:
            os.remove(self.out_video_file)
        except:
            pass

        self.save_video_settings()

    def save_video_settings(self):

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.out_video_file, fourcc,
                                            self.out_video_fps,
                                            (self.out_video_width, self.out_video_height))

    def run_best_frame_selection(self):
        print('GO')

    def srceen4biometry_mk(self, cv_img, dt):
        # drowing on screen for biometry
        srceen = Srceen(cv_img)

        srceen4biometry = srceen.process_frame(dt)

        return srceen4biometry


    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)

        #self.dt = self.timer.return_time()

        while True:
            ret, cv_img_in = cap.read()

            if ret:

                #load known faces from DB
                if self.load_DB_flag:
                    print('loading known_face_encodings')
                    self.known_face_encodings = self.db_from_file.known_face_encodings
                    self.known_face_metadata = self.db_from_file.known_face_metadata
                    print('N Users=%i' % len(self.known_face_metadata))

                    #print('known_face_encodings:')
                    #print(self.known_face_encodings)
                    #print('known_face_metadata')
                    #print(self.known_face_metadata)

                    self.load_DB_flag = False

                if self.save_in_DB_flag:
                    try:
                        userID = len(self.known_face_metadata) + 1
                        new_face_encodings, new_face_image = face_detection.extract_new_face(cv_img_in)
                        self.db_from_file.register_new_face(new_face_encodings, new_face_image, userID)
                        self.db_from_file.save_known_faces()

                        print('new userID: %s' % userID)

                    except:
                        print('ERROR in module Register New Face')

                    self.save_in_DB_flag = False
                    self.load_DB_flag = True

                ######### record video ##########
                #if self.record_video_flag:
                #    #for first frime in series:
                #    if self.timer.t_start == 0:
                #        self.first_frame_in_series()
                #
                #    ################## save video ############
                #    #self.record_video(cv_img_in)
                #
                #
                #self.video_is_recorded_flag = False
                #############################

                ########## connection with db #############
                in_to_from_process_openCV = {'known_face_encodings': self.known_face_encodings,
                                             'known_face_metadata': self.known_face_metadata,
                                             'dt': self.dt,
                                             'identified_metadata': self.identified_metadata}

                #if self.DB_connection_flag: ...

                cv_img_out, out_from_process_openCV = self._state.process_openCV_frame(cv_img_in,
                                                                                       in_to_from_process_openCV)

                self.save_in_DB_flag = out_from_process_openCV['save_in_DB_flag']

                if 'identified_metadata' in out_from_process_openCV.keys():
                    self.identified_metadata = out_from_process_openCV['identified_metadata']

                #############################


                if 'first_fime_flag' in out_from_process_openCV.keys():
                    if out_from_process_openCV['first_fime_flag']:
                        self.timer.start_timer()
                        print('start_timer')
                        #else:
                        #    print('stop_timer')
                        #    self.timer.stop_timer()

                    self.dt = self.timer.return_time()





                # emit frame to show
                self.change_pixmap_signal.emit(cv_img_out)

                # emit state
                #self.emit_state.emit(self._state.name)
                #print(self._state.name)


                ###

                ###

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        # shut down capture system
        cap.release()
        cv2.destroyAllWindows()


class MainWindow(QMainWindow, lib.SetSettings):
    '''
    класс для реализации главного окна QWidget
    '''

    def __init__(self, state: State) -> None:
        super().__init__()

        #Default mode is BackgroundMode
        self._state = state

        self.initUI()

        self.show()


    def initUI(self):

        self.set_window()

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        self.init_thread()

        #self.btn_DB = self.DB_button()

        self.btn_reg = self.registration_button()


    def registration_button(self):

        btn_reg = QPushButton(self)
        btn_reg.setText('Add new user')
        btn_reg.move(200, 370)
        btn_reg.resize(200, 100)

        btn_reg.clicked.connect(self.start_registration)

        return btn_reg

    def registration_button_go(self):
        self.btn_reg.setText('Add new user')
        self._state.change_state(FaceIdentificationMode)
        self.btn_reg.clicked.connect(self.start_registration)


    def start_registration(self):
        #self._state.change_state(UserRegistrationMode1)
        #self.btn_reg.setText('Start record')
        #self.btn_reg.clicked.connect(self.record_video)
        #self.btn_reg.clicked.connect(self._state.change_state_to_UserRegistrationMode2)
        #self.btn_reg.clicked.disconnect(self.start_registration)

        self._state.change_state(UserRegistrationMode1_quality_aware)
        self.btn_reg.setText('EXIT')

        print('start_registration')

        self.btn_reg.clicked.disconnect(self.start_registration)
        self.btn_reg.clicked.connect(self.registration_button_go)



    def record_video(self):

        print('recording video...')
        self.btn_reg.setText('EXIT')


        self.thread.record_video_flag = True

        self.btn_reg.clicked.disconnect(self.record_video)
        self.btn_reg.clicked.connect(self.registration_button_go)

        #self.btn_reg.setText('Exit')
        #self.btn_reg.clicked.connect(self._state.FaceIdentificationMode)




        #self.ss_video.clicked.connect(self.start_registration)


    #######################################



    ########################################################
    def set_window(self):
        self.setWindowTitle("FaceID")
        self.setGeometry(300, 300, self.display_width, self.display_height)

    def init_thread(self):
        ################
        # create the video capture thread
        self.thread = VideoThread(self._state)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        self.thread.emit_start_best_frame_selection_flag.connect(self.run_best_frame_sellection)
        # print(video_is_recorded_flag)

        #self.thread.emit_state.connect(self.change_state_main_window)

        # start the thread
        self.thread.start()
        ###################

    def init_text_on_screen_app(self, text):
        # create a text label
        self.layout = QVBoxLayout()
        self.label = QLabel(text)

        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    ###########################################

    ###################################################
    def DB_button(self):
        btn_DB = QPushButton(self)
        btn_DB.clicked.connect(self.disconnect_with_DB_button)
        btn_DB.setText('DB ON')
        self.DB_connection_flag = True

        return btn_DB

    def connect_with_DB_button(self):
        self.btn_DB.setText('DB is ON')
        print('DB is connected')
        self.DB_connection_flag = False

        self.btn_DB.clicked.connect(self.DB_connection)

        self.btn_DB.clicked.disconnect(self.connect_with_DB_button)
        self.btn_DB.clicked.connect(self.disconnect_with_DB_button)

    def disconnect_with_DB_button(self):
        self.btn_DB.setText('DB is OFF')
        print('DB is disconnected')
        self.DB_connection_flag = True

        self.btn_DB.clicked.connect(self.DB_connection)

        self.btn_DB.clicked.disconnect(self.disconnect_with_DB_button)
        self.btn_DB.clicked.connect(self.connect_with_DB_button)

    ###############################

    @pyqtSlot(bool)
    def DB_connection(self):
        self.thread.DB_connection_flag = self.DB_connection_flag


    @pyqtSlot(bool)
    def run_best_frame_sellection(self, start_flag):
        if start_flag:
            print('the best frame selection...')
            self.thread.runing_best_frame_sellection = True

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




if __name__ == "__main__":

    app = QApplication(sys.argv)
    mw = MainWindow(BackgroundMode())
    #mw = MainWindow(FaceIdentificationMode())
    #mw = MainWindow(GreetingsMode())
    #mw = MainWindow(UserRegistrationMode())
    sys.exit(app.exec_())



