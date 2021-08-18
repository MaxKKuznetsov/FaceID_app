import sys

from PyQt5.QtCore import Qt
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTableView,
)


class Users(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("QTableView Example")
        self.resize(1200, 800)

        # Set up the model
        self.model = QSqlTableModel(self)
        self.model.setTable("users")
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)
        self.model.setHeaderData(0, Qt.Horizontal, "id")
        self.model.setHeaderData(1, Qt.Horizontal, "name")
        self.model.setHeaderData(2, Qt.Horizontal, "face_image")
        self.model.setHeaderData(3, Qt.Horizontal, "face_encoding")
        self.model.select()

        # Set up the view
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.resizeColumnsToContents()
        self.setCentralWidget(self.view)


def createConnection():
    con = QSqlDatabase.addDatabase("QSQLITE")
    con.setDatabaseName("test_db.sqlite")
    if not con.open():
        QMessageBox.critical(
            None,
            "QTableView Example - Error!",
            "Database Error: %s" % con.lastError().databaseText(),
        )
        return False
    return True


def add_user():
    con = QSqlDatabase.addDatabase("QSQLITE")
    con.setDatabaseName("test_db.sqlite")
    con.open()

    name = "Linda"
    face_image = [1, 2, 3]
    face_encoding = [10, 20, 30]

    query = QSqlQuery()
    query.exec(
        f"""INSERT INTO users (name, face_image, face_encoding) 
        VALUES ('{name}', '{face_image}', '{face_encoding}')"""
    )

    con.close()


app = QApplication(sys.argv)

if not createConnection():
    sys.exit(1)

win = Users()
win.show()
sys.exit(app.exec_())
