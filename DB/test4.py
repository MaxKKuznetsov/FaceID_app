import sqlite3
import numpy as np

con = sqlite3.connect("test_db.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()

x = np.arange(12).reshape(2, 6)

cur.execute("insert into test (arr) values (?)", (x, ))