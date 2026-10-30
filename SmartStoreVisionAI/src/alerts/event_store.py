"""Scoped, searchable 30-day event timeline with durable acknowledgements."""
import json
import sqlite3
import time
from pathlib import Path


class EventStore:
    def __init__(self, path, organization_id='local', store_id='main-store', retention_days=30):
        self.path=str(path)
        self.organization_id=organization_id
        self.store_id=store_id
        self.retention_days=retention_days
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, organization_id TEXT NOT NULL, store_id TEXT NOT NULL,
                timestamp REAL NOT NULL, camera_id TEXT, alert_type TEXT, severity_score INTEGER,
                payload TEXT NOT NULL, acknowledged_at REAL)''')
            db.execute('CREATE INDEX IF NOT EXISTS events_scope_time ON events(organization_id,store_id,timestamp)')
        self.prune()

    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        return db

    def prune(self):
        with self.connect() as db:
            db.execute('DELETE FROM events WHERE organization_id=? AND store_id=? AND timestamp<?',
                       (self.organization_id,self.store_id,time.time()-self.retention_days*86400))

    def save(self, event):
        with self.connect() as db:
            db.execute('INSERT INTO events(id,organization_id,store_id,timestamp,camera_id,alert_type,severity_score,payload) VALUES(?,?,?,?,?,?,?,?)',
                       (event['alert_id'],self.organization_id,self.store_id,event['timestamp'],event['camera_id'],event['alert_type'],event['severity_score'],json.dumps(event,default=str)))
        self.prune()

    def query(self, since=None, until=None, camera_id=None, alert_type=None, min_severity=0, min_confidence=0., include_acknowledged=True, search='', limit=1000):
        sql='SELECT payload,acknowledged_at FROM events WHERE organization_id=? AND store_id=? AND timestamp>=?'
        params=[self.organization_id,self.store_id,max(since or 0,time.time()-self.retention_days*86400)]
        if until is not None:sql+=' AND timestamp<=?';params.append(until)
        if camera_id:sql+=' AND camera_id=?';params.append(camera_id)
        if alert_type:sql+=' AND alert_type=?';params.append(alert_type)
        sql+=' AND severity_score>=?';params.append(min_severity)
        if not include_acknowledged:sql+=' AND acknowledged_at IS NULL'
        sql+=' ORDER BY timestamp DESC'
        # Filter confidence and search before applying the limit.
        result=[]
        with self.connect() as db:
            for row in db.execute(sql,params):
                event=json.loads(row['payload'])
                if event.get('confidence',0)<min_confidence:continue
                if search and search.casefold() not in (event.get('message','')+' '+(event.get('zone_id') or '')).casefold():continue
                result.append({**event,'acknowledged_at':row['acknowledged_at']})
                if len(result)>=max(1,min(limit,10000)):break
        return result

    def acknowledge(self, event_id):
        with self.connect() as db:
            cursor=db.execute('UPDATE events SET acknowledged_at=COALESCE(acknowledged_at,?) WHERE id=? AND organization_id=? AND store_id=?',
                              (time.time(),event_id,self.organization_id,self.store_id))
            return cursor.rowcount==1
