
class Model:
    """
    Класс Model представляет собой реализацию модели данных.
    В модели хранятся..
    Модель предоставляет интерфейс, через который можно работать
    с хранимыми значениями.

    Модель содержит методы регистрации, удаления и оповещения
    наблюдателей.
    """

    def __init__(self):

        self.state = 'BackgroundMode'

        self.faces = []

        self.button_set = {'button_show_flag': True,
                            'button_text': 'Add new user',
                            'button_location': [],
                            'button_size': [],
                            }

        # список наблюдателей
        self._mObservers = []

    def return_button_set(self, InState):

        if (InState == 'BackgroundMode') or (InState == 'FaceIdentificationMode'):
            self.button_set = {'button_show_flag': True,
                                'button_text': 'Add new user',
                                'button_location': (),
                                'button_size': (),
                                }

        elif (InState == 'UserRegistrationMode') or (InState == 'GreetingsMode'):
            self.button_set = {'button_show_flag': True,
                                'button_text': 'EXIT',
                                'button_location': (),
                                'button_size': (),
                                }

        else:
            self.button_set = {'button_show_flag': False,
                                'button_text': '0',
                                'button_location': [],
                                'button_size': [],
                                }


    @property
    def change_state(self):
        return self.state, self.button_set

    @change_state.setter
    def change_state(self, InState):
        self.state = InState
        self.return_button_set(InState)

        self.notifyObservers()


    def addObserver(self, inObserver):
        self._mObservers.append(inObserver)

    def removeObserver(self, inObserver):
        self._mObservers.remove(inObserver)

    def notifyObservers(self):
        for x in self._mObservers:
            x.modelIsChanged()
