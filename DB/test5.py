from PyQt5.QtSql import QSqlQuery, QSqlDatabase

import numpy as np

def adapt_array(arr):
    return arr.astype('float64').tobytes()

def convert_array(text):
    return np.frombuffer(text, dtype='float64')

con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("test_db.sqlite")
con.open()

''' #create table
# Create a query and execute it right away using .exec()
createTableQuery = QSqlQuery()
createTableQuery.exec(
    """
    CREATE TABLE contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
        name VARCHAR(40) NOT NULL,
        job VARCHAR(50),
        email VARCHAR(40) NOT NULL
    )
    """
)
'''




# Creating a query for later execution using .prepare()
insertDataQuery = QSqlQuery()
insertDataQuery.prepare(
    """
    INSERT INTO contacts (
        name,
        job,
        email
    )
    VALUES (?, ?, ?)
    """
)

# Sample data
data = [
    ("Joe", "Senior Web Developer", "joe@example.com"),
    ("Lara", "Project Manager", "lara@example.com"),
    ("David", "Data Analyst", "david@example.com"),
    ("Jane", "Senior Python Developer", "jane@example.com"),
]

# Use .addBindValue() to insert data
for name, job, email in data:
    insertDataQuery.addBindValue(name)
    insertDataQuery.addBindValue(job)
    insertDataQuery.addBindValue(email)
    insertDataQuery.exec()

