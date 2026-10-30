"""Transactional local retail store. Amounts are stored in integer paisa."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4


class Commerce:
    transitions = {'received': ['accepted', 'cancelled'], 'accepted': ['picking', 'cancelled'],
                   'picking': ['packed', 'cancelled'], 'packed': ['ready', 'cancelled'],
                   'ready': ['dispatched', 'completed', 'cancelled'], 'dispatched': ['completed', 'failed'],
                   'failed': ['dispatched', 'cancelled'], 'completed': [], 'cancelled': []}

    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS products(org TEXT, sku TEXT, name TEXT NOT NULL, price INTEGER NOT NULL,
                cost INTEGER NOT NULL, PRIMARY KEY(org,sku));
            CREATE TABLE IF NOT EXISTS movements(id INTEGER PRIMARY KEY, org TEXT, branch TEXT, sku TEXT,
                quantity INTEGER NOT NULL, reason TEXT, reference TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(org,branch,sku,reference));
            CREATE TABLE IF NOT EXISTS customers(org TEXT, customer TEXT, name TEXT, PRIMARY KEY(org,customer));
            CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY, org TEXT, branch TEXT, source TEXT,
                customer TEXT, items TEXT, total INTEGER, payment TEXT, status TEXT, rider TEXT DEFAULT '',
                proof TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS loyalty(id INTEGER PRIMARY KEY, org TEXT, customer TEXT, points INTEGER, reference TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, org TEXT, branch TEXT, action TEXT,
                reference TEXT, detail TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TRIGGER IF NOT EXISTS no_movement_update BEFORE UPDATE ON movements
                BEGIN SELECT RAISE(ABORT, 'Append a correcting movement'); END;
            CREATE TRIGGER IF NOT EXISTS no_movement_delete BEFORE DELETE ON movements
                BEGIN SELECT RAISE(ABORT, 'Append a correcting movement'); END;
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def scope(org, branch=None):
        if not isinstance(org, str) or not org.strip() or (branch is not None and (not isinstance(branch, str) or not branch.strip())):
            raise ValueError('Organization and branch are required')

    def product(self, org, sku, name, price, cost):
        self.scope(org)
        if not sku.strip() or not name.strip() or not isinstance(price, int) or not isinstance(cost, int) or min(price,cost) < 0:
            raise ValueError('Product identifiers and nonnegative integer paisa amounts are required')
        with self.connect() as db:
            db.execute('INSERT INTO products VALUES(?,?,?,?,?) ON CONFLICT(org,sku) DO UPDATE SET name=excluded.name,price=excluded.price,cost=excluded.cost', (org,sku,name,price,cost))

    def customer(self, org, customer, name):
        self.scope(org)
        if not customer.strip() or not name.strip():
            raise ValueError('Customer ID and name required')
        with self.connect() as db:
            db.execute('INSERT INTO customers VALUES(?,?,?) ON CONFLICT(org,customer) DO UPDATE SET name=excluded.name', (org,customer,name))

    def _stock(self, db, org, branch, sku):
        return db.execute('SELECT COALESCE(SUM(quantity),0) FROM movements WHERE org=? AND branch=? AND sku=?', (org,branch,sku)).fetchone()[0]

    def _reserved(self, db, org, branch, sku):
        rows = db.execute("SELECT items FROM orders WHERE org=? AND branch=? AND status NOT IN ('completed','cancelled')", (org,branch))
        return sum(i['quantity'] for r in rows for i in json.loads(r['items']) if i['sku'] == sku)

    def stock(self, org, branch, sku):
        with self.connect() as db:
            on_hand = self._stock(db,org,branch,sku)
            reserved = self._reserved(db,org,branch,sku)
            return {'on_hand': on_hand, 'reserved': reserved, 'available': on_hand-reserved}

    def movement(self, org, branch, sku, quantity, reason, reference):
        self.scope(org,branch)
        if not isinstance(quantity,int) or quantity == 0 or not reason.strip() or not reference.strip():
            raise ValueError('Nonzero integer quantity, reason and reference required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM products WHERE org=? AND sku=?',(org,sku)).fetchone():
                raise ValueError('Unknown product')
            old = db.execute('SELECT quantity,reason FROM movements WHERE org=? AND branch=? AND sku=? AND reference=?',(org,branch,sku,reference)).fetchone()
            if old:
                if tuple(old) != (quantity,reason):
                    raise ValueError('Reference already used for a different movement')
                return
            if self._stock(db,org,branch,sku) + quantity < self._reserved(db,org,branch,sku):
                raise ValueError('Insufficient unreserved stock')
            db.execute('INSERT INTO movements(org,branch,sku,quantity,reason,reference) VALUES(?,?,?,?,?,?)',(org,branch,sku,quantity,reason,reference))

    def checkout(self, org, branch, cart, source='pos', customer='', payment='cash', reference=None):
        self.scope(org,branch)
        if source not in ('pos','online','telephone') or payment not in ('cash','external_paid','cod'):
            raise ValueError('Unsupported source or payment method')
        if source == 'pos' and payment == 'cod':
            raise ValueError('POS checkout requires payment confirmation')
        if not cart or any(not isinstance(q,int) or q <= 0 for q in cart.values()):
            raise ValueError('Cart requires positive integer quantities')
        order_id = reference or str(uuid4())
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM orders WHERE id=?',(order_id,)).fetchone()
            if old:
                old_cart = {i['sku']: i['quantity'] for i in json.loads(old['items'])}
                if (old['org'],old['branch'],old['source'],old['customer'],old['payment'],old_cart) != (org,branch,source,customer,payment,cart):
                    raise ValueError('Checkout reference already used')
                return order_id
            if customer and not db.execute('SELECT 1 FROM customers WHERE org=? AND customer=?',(org,customer)).fetchone():
                raise ValueError('Register the customer first')
            items=[]
            for sku,quantity in cart.items():
                p=db.execute('SELECT * FROM products WHERE org=? AND sku=?',(org,sku)).fetchone()
                if not p or self._stock(db,org,branch,sku)-self._reserved(db,org,branch,sku) < quantity:
                    raise ValueError(f'Insufficient available stock for {sku}')
                items.append({'sku':sku,'name':p['name'],'quantity':quantity,'price':p['price']})
            total=sum(i['price']*i['quantity'] for i in items)
            status='completed' if source=='pos' else 'received'
            db.execute('INSERT INTO orders(id,org,branch,source,customer,items,total,payment,status) VALUES(?,?,?,?,?,?,?,?,?)',(order_id,org,branch,source,customer,json.dumps(items),total,payment,status))
            if status=='completed':
                self._fulfill(db,org,branch,order_id,items,customer,total)
            db.execute('INSERT INTO audit(org,branch,action,reference,detail) VALUES(?,?,?,?,?)',(org,branch,'order_created',order_id,status))
        return order_id

    def _fulfill(self, db, org, branch, order_id, items, customer, total):
        for i in items:
            db.execute('INSERT INTO movements(org,branch,sku,quantity,reason,reference) VALUES(?,?,?,?,?,?)',(org,branch,i['sku'],-i['quantity'],'sale',order_id))
        if customer:
            db.execute('INSERT INTO loyalty(org,customer,points,reference) VALUES(?,?,?,?)',(org,customer,total//10000,order_id))

    def transition(self, org, branch, order_id, status, rider='', proof='', cod_collected=False):
        self.scope(org,branch)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM orders WHERE id=? AND org=? AND branch=?',(order_id,org,branch)).fetchone()
            if not row or status not in self.transitions[row['status']]:
                raise ValueError('Invalid order transition')
            if status=='dispatched' and not rider.strip():
                raise ValueError('Assign a rider before dispatch')
            if status=='completed':
                if row['status']=='dispatched' and not proof.strip():
                    raise ValueError('Delivery proof is required')
                if row['payment']=='cod' and not cod_collected:
                    raise ValueError('Confirm cash collection before completing this order')
                self._fulfill(db,org,branch,order_id,json.loads(row['items']),row['customer'],row['total'])
            db.execute('UPDATE orders SET status=?,rider=?,proof=? WHERE id=?',(status,rider or row['rider'],proof or row['proof'],order_id))
            db.execute('INSERT INTO audit(org,branch,action,reference,detail) VALUES(?,?,?,?,?)',(org,branch,'order_status',order_id,status))

    def rows(self, table, org, branch=None):
        if table not in ('products','orders','customers','loyalty','movements','audit'):
            raise ValueError('Unknown table')
        self.scope(org,branch)
        sql=f'SELECT * FROM {table} WHERE org=?'
        params=[org]
        if table in ('orders','movements','audit'):
            if branch is None:
                raise ValueError('Branch is required')
            sql+=' AND branch=?';params.append(branch)
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql,params)]

    def sales(self, org, branch):
        import pandas as pd
        rows=[]
        for order in self.rows('orders',org,branch):
            if order['status']=='completed':
                for i in json.loads(order['items']):
                    rows.append({'date':order['created_at'][:10], 'sku':i['sku'], 'store_id':branch,
                                 'sales_qty':i['quantity'], 'unit_price':i['price']/100, 'promotion_flag':0})
        return pd.DataFrame(rows,columns=['date','sku','store_id','sales_qty','unit_price','promotion_flag'])
