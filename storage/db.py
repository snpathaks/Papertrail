import sqlite3, json, hashlib
import config

def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    return c

def make_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()

def cache_get(key):
    with _conn() as c:
        row = c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None

def cache_put(key, value):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))