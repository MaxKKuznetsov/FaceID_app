import time
from datetime import datetime

from Utility.MainWinObserver import MainWinObserver
from Utility.MainWinMeta import MainWinMeta

def elapsed(func):
    def wrapper():
        start = datetime.now()
        func()
        end = datetime.now()
        elapsed = (end - start).total_seconds() * 1000000
        print(f'>> функция {func.__name__} время выполнения (mks): {elapsed}')

    return wrapper

def elapsed_1arg(func):
    def wrapper(arg):
        start = datetime.now()
        func(arg)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция {func.__name__} время выполнения (s): {elapsed}')

    return wrapper

def elapsed_2arg(func: object) -> object:
    """

    :rtype: object
    """
    def wrapper(arg1, arg2):
        start = datetime.now()
        func(arg1, arg2)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция {func.__name__} время выполнения (mks): {elapsed}')

    return wrapper

def elapsed_3arg(func):
    def wrapper(arg1, arg2, arg3):
        start = datetime.now()
        func(arg1, arg2, arg3)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция {func.__name__} время выполнения (mks): {elapsed}')

    return wrapper

def timer4detector(detector, detect_fn, images, *args):
    start = time.time()
    faces = detect_fn(detector, images, *args)
    elapsed = time.time() - start
    print(f', {elapsed:.3f} seconds')
    return faces, elapsed

class Timer(MainWinObserver, metaclass=MainWinMeta):
    def __init__(self, inModel):
        """
        :type inModel: object
        """
        self.start_timer()

        # self.start_timer()
        # self.dt = 0

        self.mModel = inModel
        self.modelIsChanged()

        # регистрируем представление в качестве наблюдателя
        self.mModel.addObserver(self)

    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """
        # self.return_time()
        # print(self.mModel.state)
        # print('time end: %s' % str(self.return_time()))
        self.stop_timer()
        self.start_timer()

    def start_timer(self):
        self.t_start = time.time()

    def stop_timer(self):
        self.t_start = 0

    def return_time(self):
        return time.time() - self.t_start
