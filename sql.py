import sqlite3
from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS ads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    thread_id INTEGER,
    text TEXT,
    photo_id TEXT,
    timestamp INTEGER
)
""")

# Таблица для предупреждений с колонкой ad_key
cursor.execute("""
CREATE TABLE IF NOT EXISTS warnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    ad_key TEXT,
    warning_count INTEGER,
    last_warning INTEGER
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS topics (
    thread_id INTEGER PRIMARY KEY,
    enabled INTEGER DEFAULT 1,
    block_days INTEGER DEFAULT 5,
    warnings_limit INTEGER DEFAULT 3,
    ad_frequency_days INTEGER DEFAULT 5
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    first_name TEXT,
    banned_until INTEGER,
    reason TEXT DEFAULT 'Не указано'
)
""")

conn.commit()
conn.close()
