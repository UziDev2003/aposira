"""Append-only stock movement ledger, scoped to organization, branch and SKU."""
import math
from contextlib import contextmanager
import sqlite3
from pathlib import Path


class StockLedger:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS stock_movements (
                id INTEGER PRIMARY KEY, organization_id TEXT NOT NULL, store_id TEXT NOT NULL,
                sku TEXT NOT NULL, quantity REAL NOT NULL, reason TEXT NOT NULL, reference TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(organization_id, store_id, reference))''')
            for action in ('UPDATE', 'DELETE'):
                db.execute(f'''CREATE TRIGGER IF NOT EXISTS stock_no_{action.lower()} BEFORE {action} ON stock_movements
                    BEGIN SELECT RAISE(ABORT, 'Stock movements are immutable; add a correcting movement'); END''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, organization_id, store_id, sku, quantity, reason, reference):
        if not all(isinstance(v, str) and v.strip() for v in (organization_id, store_id, sku, reason, reference)):
            raise ValueError('Organization, store, SKU, reason and unique reference are required')
        if not math.isfinite(quantity) or quantity == 0:
            raise ValueError('Movement quantity must be finite and nonzero')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT sku, quantity, reason FROM stock_movements WHERE organization_id=? AND store_id=? AND reference=?', (organization_id, store_id, reference)).fetchone()
            if old:
                if old != (sku, quantity, reason):
                    raise ValueError('Reference already belongs to a different movement')
                return False
            balance = db.execute('SELECT COALESCE(SUM(quantity),0) FROM stock_movements WHERE organization_id=? AND store_id=? AND sku=?', (organization_id, store_id, sku)).fetchone()[0]
            if balance + quantity < 0:
                raise ValueError('Movement would create negative stock')
            db.execute('INSERT INTO stock_movements (organization_id,store_id,sku,quantity,reason,reference) VALUES (?,?,?,?,?,?)', (organization_id,store_id,sku,quantity,reason,reference))
        return True

    def balance(self, organization_id, store_id, sku):
        with self.connect() as db:
            row = db.execute('SELECT COUNT(*), COALESCE(SUM(quantity),0) FROM stock_movements WHERE organization_id=? AND store_id=? AND sku=?', (organization_id,store_id,sku)).fetchone()
        return row[1] if row[0] else None

    def history(self, organization_id, store_id):
        import pandas as pd
        with self.connect() as db:
            return pd.read_sql('SELECT * FROM stock_movements WHERE organization_id=? AND store_id=? ORDER BY id DESC', db, params=(organization_id,store_id))
