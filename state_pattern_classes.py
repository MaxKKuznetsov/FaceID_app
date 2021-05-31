#from __future__ import annotations # do not work for python V<3.7!
from abc import ABC, abstractmethod


class State(ABC):
    """
    Базовый класс Состояния объявляет методы, которые должны реализовать все
    Конкретные Состояния, а также предоставляет обратную ссылку на объект
    Контекст, связанный с Состоянием. Эта обратная ссылка может использоваться
    Состояниями для передачи Контекста другому Состоянию.
    """

    @abstractmethod
    def handle12(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def handle21(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def handle23(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def handle32(self) -> None:
        raise NotImplementedError()


class BackgroundMode(State):

    def __init__(self):
        pass


    def handle12(self) -> None:
        return 'BackgroundMode(1) --> handle12'

    def handle21(self) -> None:
        pass

    def handle23(self) -> None:
        pass

    def handle32(self) -> None:
        pass


class FaceIdentificationMode(State):

    def __init__(self):
        pass

    def handle12(self) -> None:
        return 'FaceIdentificationMode(2) --> handle12'

    def handle21(self) -> None:
        pass


    def handle23(self) -> None:
        pass

    def handle32(self) -> None:
        pass


class UserRegisrationMode(State):

    def __init__(self):
        pass


    def handle12(self) -> None:
        return 'UserRegisrationMode(3) --> handle12'

    def handle21(self) -> None:
        pass

    def handle23(self) -> None:
        pass

    def handle32(self) -> None:
        pass



class App:

    def __init__(self, state: State) -> None:
        self._state = state


    def test_input(self):
        while True:
            print('input something')
            self.key = input()

            if self.key == '1':
                self.change_state(background_mode)

            elif self.key == '2':
                self.change_state(face_identification_mode)

            elif self.key == '3':
                self.change_state(user_regisration_mode)

            out = self.handle12()

            if self.key == 'q':
                break

    def change_state(self, state: State) -> None:
        self._state = state

    def handle12(self) -> None:
        self._execute('handle12')



    def _execute(self, operation: str) -> None:
        try:
            func = getattr(self._state, operation)
            print('App {}.'.format(func()))
        except AttributeError:
            print('our App can not do this --> kill all humans mode in ON')


if __name__ == "__main__":
    # Клиентский код.
    background_mode = BackgroundMode()
    face_identification_mode = FaceIdentificationMode()
    user_regisration_mode = UserRegisrationMode()

    a = App(background_mode)
    print('##########GO##############')

    a.test_input()

    #app.change_state(face_identification_mode)
    #app.handle21()



    #context = Context(BackgroundMode())
    #context.request12()
