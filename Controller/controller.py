from View.view import View
from Utility.timer import Timer

from Utility.timer import elapsed_1arg, elapsed_2arg, elapsed_3arg

class Controller:
    """
    Класс Controller представляет реализацию контроллера.
    Согласовывает работу представления с моделью.
    """

    def __init__(self, inModel):
        """
        Конструктор принимает ссылку на модель.
        Конструктор создаёт и отображает представление.
        """

        # Load data processing class
        self.mModel = inModel

        self._state = self.mModel.state

        self.timer = Timer(self.mModel)

        # Load ui interface class, used to set ui
        self.mView = View(self, self.mModel)

        # button from app
        self.mView.btn_reg.clicked.connect(self.controler_change_state)

        #check_face_size
        self.mView.thread.check_face_size.connect(self.face_size_change_state)

        #Face identidication
        self.mView.thread.metadatas_out.connect(self.change_state_to_GreetingsMode)

        ### Out from GreetingsMode
        self.mView.thread.timer_time.connect(self.change_state_outOf_GreetingsMode)

        ### Out from AlreadyRegistered
        self.mView.thread.timer_time.connect(self.change_state_outOf_AlreadyRegistered)

        ### Save new user
        self.mView.thread.emit_face_quality_limit.connect(self.save_new_user)

        ### Out from save new user
        self.mView.thread.timer_time.connect(self.save_new_user_out)

        self.mView.show()


    def save_new_user(self, face_quality_limit_flag):
        if face_quality_limit_flag:
            self._state = 'SaveNewUserMode'
            self.mModel.change_state = self._state

    def save_new_user_out(self, time):
        if (self._state == 'SaveNewUserMode') and time > 7:
            self._state = 'FaceIdentificationMode'
            self.mModel.change_state = self._state

    def change_state_outOf_GreetingsMode(self, time):
        if self._state == 'GreetingsMode':
            #print(time)
            if time > 5:
                self._state = 'FaceIdentificationMode'
                self.mModel.change_state = self._state

    def change_state_outOf_AlreadyRegistered(self, time):
        if self._state == 'AlreadyRegistered':
            #print(time)
            if time > 2:
                self._state = 'FaceIdentificationMode'
                self.mModel.change_state = self._state

    def change_state_to_GreetingsMode(self, metadatas):

        if not self._state == 'GreetingsMode':
            if metadatas:
                if self._state == 'UserRegistrationMode':
                    self._state = 'AlreadyRegistered'

                elif self._state == 'FaceIdentificationMode':
                    self._state = 'GreetingsMode'

                else:
                    self._state = 'GreetingsMode'

            self.mModel.change_state = self._state


    def face_size_change_state(self, face_size_flag):

        # print(face_size_flag)

        if not face_size_flag and (self._state == 'FaceIdentificationMode'):
            self._state = 'BackgroundMode'
            self.mModel.change_state = self._state

        elif face_size_flag and (self._state == 'BackgroundMode'):
            self._state = 'FaceIdentificationMode'
            self.mModel.change_state = self._state

    def controler_change_state(self):

        if (self._state == 'BackgroundMode') or (self._state == 'FaceIdentificationMode'):
            self._state = 'UserRegistrationMode'

        else:
            self._state = 'BackgroundMode'

        self.mModel.change_state = self._state
