import pickle
import numpy as np
import cv2
import sys

from PyQt5.QtSql import QSqlQuery, QSqlDatabase

con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("test_db.sqlite")
con.open()

name = "Bob"
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[11, 21, 31], [41, 51, 61]])


def adapt_array(arr):
    return arr.astype('float64').tobytes()

def convert_array(text):
    return np.frombuffer(text, dtype='float64')

shape_in1 = array1.shape
print(shape_in1)

array1 = adapt_array(array1)
print(type(array1))

#
array2 = array2.tostring()
print(array2)
print(type(array2))

#test1 = pickle.dumps(new_face_encoding)


query = QSqlQuery(con)
query.exec(
    f"""INSERT INTO users (name, face_image, face_encoding, blob) 
    VALUES ('{name}', '{11}', '{111}', {111})"""
)

#test2 = convert_array(test1)
#print(test2)

#print(test2.reshape(shape_in))


'''
#write user
query = QSqlQuery(con)
query.exec(
    f"""INSERT INTO users (name, face_image, face_encoding) 
    VALUES ('{name}', '{[new_face_image]}', '{new_face_encoding}')"""
)
'''


'''
def show_img(face_image) -> object:
    img = cv2.imread(face_image)

    if img is None:
        sys.exit("Could not read the image.")

    cv2.imshow("Display window", img)

    k = cv2.waitKey(0)

'''
'''
# read db
query = QSqlQuery()

query1 = 'SELECT name, face_image, face_encoding FROM users'
query.exec()
name_ind, face_image_ind, face_encoding_ind = range(3)

while query.next():
    name, face_image, face_encoding = query.value(name_ind), query.value(face_image_ind), query.value(
        face_encoding_ind)

    # face_image = np.fromstring(face_image)
    # face_encoding = np.fromstring(face_encoding)

    print('face_encoding:')
    print(face_encoding)
    print(face_encoding[:-1])

    #show_img(face_image)

'''


# Закрываем подключение к БД
con.close()
