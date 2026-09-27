"""Exact-value request reuse across evaluator outputs (no content hashes)."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.tmp') as stream:
        name = stream.name
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def cached_call(database, identity, produce):
    # One lock per exact request: unrelated API calls remain concurrent.
    import fcntl
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=60, isolation_level=None)
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS requests '
                           '(id INTEGER PRIMARY KEY, request TEXT UNIQUE NOT NULL, result TEXT)')
        key = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        connection.execute('INSERT OR IGNORE INTO requests(request) VALUES (?)', (key,))
        row_id = connection.execute('SELECT id FROM requests WHERE request=?', (key,)).fetchone()[0]
        locks = database.with_suffix('.locks')
        locks.mkdir(exist_ok=True)
        with (locks / str(row_id)).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stored = connection.execute('SELECT result FROM requests WHERE id=?', (row_id,)).fetchone()[0]
            if stored is not None:
                return json.loads(stored), True
            result = produce()
            connection.execute('UPDATE requests SET result=? WHERE id=?',
                               (json.dumps(result, ensure_ascii=False), row_id))
            return result, False
    finally:
        connection.close()
