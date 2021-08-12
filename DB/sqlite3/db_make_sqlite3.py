import sys
from PyQt5.QtSql import QSqlDatabase, QSqlQuery

# Create the connection
con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("test_db.sqlite")

# Open the connection
if not con.open():
    print("Database Error: %s" % con.lastError().databaseText())
    sys.exit(1)

# Create a query and execute it right away using .exec()
createTableQuery = QSqlQuery()
createTableQuery.exec(
    """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
        name VARCHAR(40) NOT NULL,
        face_image VARCHAR(13000),
        face_encoding VARCHAR(200)
    )
    """
)

col = ['id', 'name', 'face_image', 'face_encoding']

print(con.tables())