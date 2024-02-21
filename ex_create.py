
import sqlite3

db_file = 'sensor_data.db'
#db_file = 'sensor_data.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

query = """
CREATE TABLE IF NOT EXISTS sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time DATETIME NOT NULL,
    temperature REAL,
    humidity REAL,
    flux1 REAL,
    flux2 REAL,
    flux3 REAL,
    flux4 REAL,
    flex REAL,
    air_quality INTEGER,
    tilt1 REAL,
    tilt2 REAL,
    tilt3 REAL,
    tilt4 REAL,
    vibe1 REAL,
    vibe2 REAL
);
"""
cursor.execute(query)

# # 인덱스 설정
# cursor.execute("CREATE INDEX idx_sensor_data_time ON sensor_data(time);")

conn.close()