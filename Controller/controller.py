from View.view import View


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

        # Load ui interface class, used to set ui
        self.mView = View(self, self.mModel)

        #button from app
        self.mView.btn_reg.clicked.connect(self.controler_change_state)

        #check face size
        self.mView.thread.check_face_size.connect(self.face_size_change_state)


        self.mView.show()

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

        elif self._state == 'UserRegistrationMode':
            self._state = 'FaceIdentificationMode'

        elif self._state == 'GreetingsMode':
            self._state = 'FaceIdentificationMode'

        self.mModel.change_state = self._state

