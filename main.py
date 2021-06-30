import sys

from PyQt5 import QtWidgets

from Model.model import Model
from Controller.controller import Controller


def main():
    app = QtWidgets.QApplication(sys.argv)

    # создаём модель
    model = Model()

    # создаём контроллер и передаём ему ссылку на модель
    controller = Controller(model)

    app.exec()


if __name__ == '__main__':
    sys.exit(main())
