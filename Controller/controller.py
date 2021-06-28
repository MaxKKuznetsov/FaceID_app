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

        #self.state_ID = 0
        #self.state_list = ['BackgroundMode',
        #                   'FaceIdentificationMode',
        #                   'GreetingsMode',
        #                   'UserRegistrationMode',
        self._state = 'BackgroundMode'

        # Load data processing class
        self.mModel = inModel

        # Load ui interface class, used to set ui
        self.mView = View(self, self.mModel)

        self.mView.btn_reg.clicked.connect(self.controler_change_state)

        self.mView.show()

    def controler_change_state(self):
        ## Use the function of model to process data
        #if self.state_ID < len(self.state_list) - 1:
        #    self.state_ID += 1
        #else:
        #    self.state_ID = 0

        if (self._state == 'BackgroundMode') or (self._state == 'FaceIdentificationMode'):
            self._state = 'UserRegistrationMode'

        elif self._state == 'UserRegistrationMode':
            self._state = 'FaceIdentificationMode'

        elif self._state == 'GreetingsMode':
            self._state = 'FaceIdentificationMode'







        self.mModel.change_state = self._state


