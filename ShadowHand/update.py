import sqlite3
import numpy as np

def update_data():
    conn = sqlite3.connect('../../../Projects/HandTailor/hand.db')
    cursor = conn.execute('SELECT * FROM pose')
    data = cursor.fetchall()
    data = np.array(data)
    return data
