import time
from datetime import datetime

from Utility.MainWinObserver import MainWinObserver
from Utility.MainWinMeta import MainWinMeta

def elapsed(func):
    def wrapper(a, b, delay=0):
        start = datetime.now()
        func(a, b, delay)
        end = datetime.now()
        elapsed = (end - start).total_seconds() * 1000
        print(f'>> функция {func.__name__} время выполнения (ms): {elapsed}')

    return wrapper


class Timer(MainWinObserver, metaclass=MainWinMeta):
    def __init__(self, inModel):
        """
        :type inModel: object
        """
        self.start_timer()

        #self.start_timer()
        #self.dt = 0

        self.mModel = inModel
        self.modelIsChanged()

        # регистрируем представление в качестве наблюдателя
        self.mModel.addObserver(self)

    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """
        #self.return_time()
        #print(self.mModel.state)
        #print('time end: %s' % str(self.return_time()))
        self.stop_timer()
        self.start_timer()

    def start_timer(self):
        self.t_start = time.time()

    def stop_timer(self):
        self.t_start = 0

    def return_time(self):
        return time.time() - self.t_start
