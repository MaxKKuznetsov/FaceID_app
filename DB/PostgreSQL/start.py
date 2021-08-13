# https://proglib.io/p/kak-podruzhit-python-i-bazy-dannyh-sql-podrobnoe-rukovodstvo-2020-02-27

import psycopg2
from psycopg2 import OperationalError


# Определим функцию create_connection() для подключения к базе данных PostgreSQL
def create_connection(db_name, db_user, db_password, db_host, db_port):

    connection = None

    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")

    except OperationalError as e:
        print(f"The error '{e}' occurred")

    return connection

def create_database(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")

connection = create_connection(
    "postgres", "postgres", "abc123", "127.0.0.1", "5432"
)

create_database_query = "CREATE DATABASE sm_app"
create_database(connection, create_database_query)

