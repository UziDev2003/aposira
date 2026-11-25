"""Durable, branch-scoped processing status and diagnostic messages."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class RunStore:
    def __init__(self, path, organization_id, store_id):
        self.path = str(path)
        self.scope = (organization_id, store_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS vision_runs (
                    id TEXT PRIMARY KEY, organization_id TEXT, store_id TEXT,
                    camera_id TEXT, source TEXT, status TEXT, started_at REAL,
                    updated_at REAL, frame INTEGER DEFAULT 0, total INTEGER DEFAULT 0,
                    summary TEXT DEFAULT '{}', error TEXT DEFAULT '');
                CREATE INDEX IF NOT EXISTS vision_runs_scope
                    ON vision_runs(organization_id, store_id, started_at);
                CREATE TABLE IF NOT EXISTS vision_run_errors (
                    id INTEGER PRIMARY KEY, run_id TEXT, timestamp REAL,
                    stage TEXT, frame INTEGER, message TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def start(self, camera_id, source):
        run_id = str(uuid4())
        now = time.time()
        with self.connect() as db:
            db.execute('''INSERT INTO vision_runs
                (id, organization_id, store_id, camera_id, source, status, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)''',
                (run_id, *self.scope, camera_id, source, now, now))
        return run_id

    def update(self, run_id, frame=0, total=0, status='running', summary=None, error=''):
        if status not in ('running', 'completed', 'failed', 'interrupted'):
            raise ValueError('Invalid processing status')
        with self.connect() as db:
            db.execute('''UPDATE vision_runs SET frame=?, total=?, status=?,
                updated_at=?, summary=?, error=?
                WHERE id=? AND organization_id=? AND store_id=?''',
                (frame, total, status, time.time(), json.dumps(summary or {}),
                 error, run_id, *self.scope))

    def error(self, run_id, stage, frame, message):
        with self.connect() as db:
            db.execute('''INSERT INTO vision_run_errors(run_id, timestamp, stage, frame, message)
                SELECT id, ?, ?, ?, ? FROM vision_runs
                WHERE id=? AND organization_id=? AND store_id=?''',
                (time.time(), stage, frame, str(message), run_id, *self.scope))

    def recent(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('''SELECT id, camera_id, source,
                status, frame, total, started_at, updated_at, error FROM vision_runs
                WHERE organization_id=? AND store_id=? ORDER BY started_at DESC LIMIT 50''',
                self.scope)]

    def errors(self, run_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute('''SELECT e.timestamp, e.stage, e.frame, e.message
                FROM vision_run_errors e JOIN vision_runs r ON r.id=e.run_id
                WHERE r.id=? AND r.organization_id=? AND r.store_id=? ORDER BY e.id DESC LIMIT 100''',
                (run_id, *self.scope))]
