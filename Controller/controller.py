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

        # button from app
        self.mView.btn_reg.clicked.connect(self.controler_change_state)

        ### Face identification
        #if state == 'UserRegistrationMode':

            # face_identification
        #    ident_limit = 0.6
        #    if self.mModel.known_face_encodings and self.mModel.known_face_metadata:
        #        metadatas = facal_processing.face_identification(cv_img_in, self.mModel.known_face_encodings,
        #                                                         self.mModel.known_face_metadata,
        #                                                         ident_limit)
        #
        #if metadatas:
        #    self.metadatas_out.emit(True)
        # else:
        #    self.metadatas_out.emit(False)

        # print(metadatas)

        # check face size
        self.mView.thread.check_face_size.connect(self.face_size_change_state)

        #Face identidication
        self.mView.thread.metadatas_out.connect(self.change_state_to_GreetingsMode)

        self.mView.show()

    def change_state_to_GreetingsMode(self, metadatas):
        if metadatas:
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

        elif self._state == 'UserRegistrationMode':
            self._state = 'FaceIdentificationMode'

        elif self._state == 'GreetingsMode':
            self._state = 'FaceIdentificationMode'

        self.mModel.change_state = self._state
