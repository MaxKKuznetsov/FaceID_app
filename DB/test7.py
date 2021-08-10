import pickle
import numpy as np
import cv2
import sys
import sqlite3

con = sqlite3.connect("data.db")
cursor = con.cursor()

id_list = [1, 2, 3]
id_tuple = tuple(id_list)

query = 'SELECT * FROM data WHERE id IN {};'.format(id_tuple)
print(query)

cursor.execute(query)