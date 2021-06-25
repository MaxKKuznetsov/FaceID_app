
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

        self._state = 'state1'

        self._buttot_set = {'button_show_flag': True,
                            'button_text': 'State1',
                            'button_location': [],
                            'button_size': [],
                            }

        # список наблюдателей
        self._mObservers = []


    def return_button_set(self, InState):

        if InState == 'state1':
            self._buttot_set = {'button_show_flag': True,
                                'button_text': 'State1',
                                'button_location': (),
                                'button_size': (),
                                }

        elif InState == 'state2':
            self._buttot_set = {'button_show_flag': True,
                                'button_text': 'State2',
                                'button_location': (),
                                'button_size': (),
                                }
        else:
            self._buttot_set = {'button_show_flag': False,
                                'button_text': '0',
                                'button_location': [],
                                'button_size': [],
                                }


    @property
    def change_state(self):
        return self._state, self._buttot_set

    @change_state.setter
    def change_state(self, InState):
        self._state = InState
        self.return_button_set(InState)

        self.notifyObservers()


    def addObserver(self, inObserver):
        self._mObservers.append(inObserver)

    def removeObserver(self, inObserver):
        self._mObservers.remove(inObserver)

    def notifyObservers(self):
        for x in self._mObservers:
            x.modelIsChanged()


