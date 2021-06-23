"""
Модуль реализации метакласса, необходимого для работы представления.

pyqtWrapperType - метакласс общий для оконных компонентов Qt.
ABCMeta - метакласс для реализации абстрактных суперклассов.

CplusDMeta - метакласс для представления.
"""

from PyQt5 import QtCore
from abc import ABCMeta

class MainWinMeta(type(QtCore.QObject), ABCMeta):
    pass