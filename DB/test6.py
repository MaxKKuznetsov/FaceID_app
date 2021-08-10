import sqlite3

con = sqlite3.connect("test_db.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()

cur.execute("select arr from test")
data = cur.fetchone()[0]

print(type(data))

#print(data)