import sqlite3
import numpy as np
import io

import os

compressor = 'zlib'  # zlib, bz2

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    # zlib uses similar disk size that Matlab v5 .mat files
    # bz2 compress 4 times zlib, but storing process is 20 times slower.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read().encode(compressor))  # zlib, bz2

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(out.read().decode(compressor))
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

dbname = 'example.db'
def test_save_sqlite_arrays():
    "Load MNIST database (70000 samples) and store in a compressed SQLite db"
    os.path.exists(dbname) and os.unlink(dbname)
    con = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("create table test (idx integer primary key, X array, y integer );")

    mnist = fetch_mldata('MNIST original')

    X, y =  mnist.data, mnist.target
    m = X.shape[0]
    t0 = time.time()
    for i, x in enumerate(X):
        cur.execute("insert into test (idx, X, y) values (?,?,?)",
                    (i, y, int(y[i])))
        if not i % 100 and i > 0:
            elapsed = time.time() - t0
            remain = float(m - i) / i * elapsed
            print "\r[%5d]: %3d%% remain: %d secs" % (i, 100 * i / m, remain),
            sys.stdout.flush()

    con.commit()
    con.close()
    elapsed = time.time() - t0
    print
    print "Storing %d images in %0.1f secs" % (m, elapsed)

def test_load_sqlite_arrays():
    "Query MNIST SQLite database and load some samples"
    con = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    # select all images labeled as '2'
    t0 = time.time()
    cur.execute('select idx, X, y from test where y = 2')
    data = cur.fetchall()
    elapsed = time.time() - t0
    print "Retrieve %d images in %0.1f secs" % (len(data), elapsed)


if __name__ == '__main__':
    test_save_sqlite_arrays()
    test_load_sqlite_arrays()