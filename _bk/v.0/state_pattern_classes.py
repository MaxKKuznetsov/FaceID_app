######
import face_detection

from abc import ABC, abstractmethod
from screen import Srceen
from face_detection import FacialImageProcessing

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QObject
import lib

class State(ABC):
    """
    Базовый класс Состояния объявляет методы, которые должны реализовать все
    Конкретные Состояния, а также предоставляет обратную ссылку на объект
    Контекст, связанный с Состоянием. Эта обратная ссылка может использоваться
    Состояниями для передачи Контекста другому Состоянию.
    """

    name = "state"
    allowed = []

    def __init__(self):
        self.first_fime = True



    def switch(self, state):
        """ Switch to new state """
        if state.name in self.allowed:
            #print('State:', self, ' => switched to new state', state.name)
            self.__class__ = state
        else:
            print('State:', self, ' => switching to', state.name, 'not possible.')

    @abstractmethod
    def text_on_screen_app(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def text_on_screen_openCV(self) -> None:
        raise NotImplementedError()



    @abstractmethod
    def process_openCV_frame(self, frame: object, in_to_from_process_openCV: object) -> object:
        raise NotImplementedError()


    @abstractmethod
    def main_window_buttons(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state_to_BackgroundMode(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state_to_FaceIdentificationMode(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state_to_UserRegistrationMode1(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state_to_UserRegistrationMode2(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state_to_UserRegistrationMode1_quality_aware(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def change_state(self, state) -> None:
        raise NotImplementedError()


    def __str__(self):
        return self.name





class BackgroundMode(State):

    name = 'BackgroundMode'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'UserRegistrationMode1_quality_aware']


    def text_on_screen_app(self) -> str:
        text = 'App: BackgroundMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'OpenCV: BackgroundMode'
        return text

    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']

        self.save_in_DB_flag = False
        

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)
        faces = facal_processing.detect_face()
        faces = facal_processing.fase_size_calc(faces)

        frame = screen.process_frame_BackgroundMode(faces)

        if facal_processing.fase_size_test(faces):
            self.change_state_to_FaceIdentificationMode()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}

        return frame, out_from_process_openCV


    def main_window_buttons(self) -> str:
        return 'BackgroundMode'

    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)



    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)

    def change_state(self, state):
        self.switch(state)




class FaceIdentificationMode(State):

    name = 'FaceIdentificationMode'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'GreetingsMode', 'UserRegistrationMode1_quality_aware']

    def text_on_screen_app(self) -> str:
        text = 'App: FaceIdentificationMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'OpenCV: FaceIdentificationMode'
        return text


    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']

        self.save_in_DB_flag = False
        

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)
        faces = facal_processing.detect_face()
        faces = facal_processing.fase_size_calc(faces)

        #face_identification
        ident_limit = 0.6
        if known_face_encodings and known_face_metadata:
            metadatas = facal_processing.face_identification(frame, known_face_encodings, known_face_metadata, ident_limit)
        else:
            metadatas = []

        #print('metadata')
        #print(metadata)

        if metadatas:
            self.change_state_to_GreetingsMode()


        screen.process_frame_FaceIdentificationMode(faces)


        if not facal_processing.fase_size_test(faces):
            self.change_state_to_BackgroundMode()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag,
                                   'identified_metadata': metadatas}

        return frame, out_from_process_openCV

    def main_window_buttons(self) -> str:
        return 'FaceIdentificationMode'


    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        pass



    def change_state_to_GreetingsMode(self):
        self.change_state(GreetingsMode)

    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)

    def change_state(self, state):
        self.switch(state)



class UserRegistrationMode1_quality_aware(State):

    name = 'UserRegistrationMode1_quality_aware'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'UserRegistrationMode1_quality_aware', 'GreetingsMode']

    def __init__(self):
        self.save_in_DB_flag1 = False


    def text_on_screen_app(self) -> str:
        text = 'App: UserRegistrationMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'OpenCV: UserRegistrationMode'
        return text


    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:

        self.save_in_DB_flag = False

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)

        faces = facal_processing.detect_face()


        if faces:
            #print(faces)
            for k in range(len(faces)):
                contrast, brightness, varianceOfLaplacian, face_confidence, FaceQuality = facal_processing.frame_quality_aware(frame, faces[k])

                faces[k]['contrast'], faces[k]['brightness'], faces[k]['focus'] = contrast, brightness, varianceOfLaplacian
                faces[k]['face_confidence'], faces[k]['FaceQuality'] = face_confidence, FaceQuality



        ###########################################
        # если лицо уже есть в базе данных:
        #face_identification
        ident_limit = 0.65
        if known_face_encodings and known_face_metadata:
            metadata = facal_processing.face_identification(frame, known_face_encodings, known_face_metadata, ident_limit)
        else:
            metadata = []

        #print('metadata:')
        #print(metadata)
        #print('--------------')

        if metadata:
            print('!!!This face is in database already!!!')


            self.save_in_DB_flag = False

            frame = screen.process_frame_UserRegistrationMode_there_is_user(faces)

            self.change_state_to_GreetingsMode2()

            out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}
            return frame, out_from_process_openCV

        #########################################

        #############################
        #оценка качества кадра для проблемы распознования
        faces = facal_processing.fase_size_calc(faces)

        face_quality = facal_processing.face_quality_limit(frame, faces)

        frame = screen.process_frame_UserRegistrationMode_quality_aware_go(faces)
        ################################

        ### если качество кадра для проблемы распознования выше значения - пишем лицо в БД
        if face_quality:
            self.save_in_DB_flag = True
            screen.sound_camera()
            self.change_state_to_FaceIdentificationMode()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}
        return frame, out_from_process_openCV



    def main_window_buttons(self) -> bool:
        return True

    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)

    def change_state_to_GreetingsMode(self):
        self.change_state(GreetingsMode)

    def change_state_to_GreetingsMode2(self):
        self.change_state(GreetingsMode2)

    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)


    def change_state_to_UserRegistrationMode1_quality_aware(self):
        pass

    def change_state(self, state):
        self.switch(state)



class GreetingsMode(State):

    name = 'GreetingsMode'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'UserRegistrationMode1_quality_aware']

    def text_on_screen_app(self) -> str:
        text = 'App: GreetingsMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'At your command, master'
        return text

    def main_window_buttons(self) -> bool:
        return False


    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:
        self.save_in_DB_flag = False

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)
        faces = facal_processing.detect_face()
        #faces = facal_processing.fase_size_calc(faces)


        #print('metadata')
        #print(in_to_from_process_openCV)


        metadatas = in_to_from_process_openCV['identified_metadata']



        frame = screen.process_frame_GreetingsMode(faces, metadatas)


        if self.first_fime:
            screen.sound_Greetings()

            self.first_fime = False


        dt_greetings = 10
        if (dt > dt_greetings) and facal_processing.fase_size_test_out(faces):
            self.first_fime = True
            self.change_state_to_BackgroundMode()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag, 'first_fime_flag': self.first_fime}

        #print('dt=%s' % dt)

        return frame, out_from_process_openCV


    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)

    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)

    def change_state(self, state):
        self.switch(state)



class GreetingsMode2(State):

    name = 'GreetingsMode'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'UserRegistrationMode1_quality_aware']

    def text_on_screen_app(self) -> str:
        text = 'App: GreetingsMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'At your command, master'
        return text

    def main_window_buttons(self) -> bool:
        return False

    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:
        self.save_in_DB_flag = False

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)
        faces = facal_processing.detect_face()
        #faces = facal_processing.fase_size_calc(faces)


        #print('metadata')
        #print(metadata)
        metadata = []


        frame = screen.process_frame_GreetingsMode2(faces, metadata)


        if self.first_fime:
            screen.sound_Greetings()
            self.first_fime = False


        if facal_processing.fase_size_test_out(faces):
            self.first_fime = True
            self.change_state_to_BackgroundMode()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}
        return frame, out_from_process_openCV


    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)

    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)

    def change_state(self, state):
        self.switch(state)

















class UserRegistrationMode1(State):

    name = 'UserRegistrationMode1'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2',
               'UserRegistrationMode1_quality_aware']


    def text_on_screen_app(self) -> str:
        text = 'App: UserRegistrationMode'
        return text

    def text_on_screen_openCV(self) -> str:
        text = 'OpenCV: UserRegistrationMode'
        return text

    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:
        self.save_in_DB_flag = False

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']
        

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)

        frame = screen.process_frame_UserRegistrationMode_step1()

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}
        return frame, out_from_process_openCV

    def main_window_buttons(self) -> bool:
        return True

    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)

    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)


    def change_state(self, state):
        self.switch(state)





class UserRegistrationMode2(State):

    name = 'UserRegistrationMode2'
    allowed = ['BackgroundMode', 'FaceIdentificationMode', 'UserRegistrationMode1', 'UserRegistrationMode2', 'UserRegistrationMode1_quality_aware']

    def process_openCV_frame(self, frame, in_to_from_process_openCV) -> list:
        self.save_in_DB_flag = False

        known_face_encodings = in_to_from_process_openCV['known_face_encodings']
        known_face_metadata = in_to_from_process_openCV['known_face_metadata']
        dt = in_to_from_process_openCV['dt']
        

        screen = Srceen(frame)
        facal_processing = FacialImageProcessing(frame, known_face_encodings, known_face_metadata)

        frame = screen.process_frame_UserRegistrationMode_step2_RecordVideo(dt)

        out_from_process_openCV = {'save_in_DB_flag': self.save_in_DB_flag}
        return frame, out_from_process_openCV

    def main_window_buttons(self) -> bool:
        return True

    def change_state_to_BackgroundMode(self):
        self.change_state(BackgroundMode)

    def change_state_to_FaceIdentificationMode(self):
        self.change_state(FaceIdentificationMode)


    def change_state_to_UserRegistrationMode1(self):
        self.change_state(UserRegistrationMode1)

    def change_state_to_UserRegistrationMode2(self):
        self.change_state(UserRegistrationMode2)

    def change_state_to_UserRegistrationMode1_quality_aware(self):
        self.change_state(UserRegistrationMode1_quality_aware)

    def change_state(self, state):
        self.switch(state)


