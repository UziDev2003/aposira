import sys
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'retail-forecasting'),str(ROOT/'SmartStoreVisionAI')]
from aposira.commerce import Commerce
from backend.core.data_validator import DataValidator
from backend.core.decision_service import forecast_sales,inventory_plan
from backend.core.optimization.simulator import InventorySimulator
from backend.core.stock_ledger import StockLedger
from backend.core.reports import export_table
from src.alerts.event_store import EventStore


def history():
    return pd.DataFrame({'date':pd.date_range('2025-01-01',periods=220),'sku':'A','store_id':'main',
                         'sales_qty':np.resize([1,2,3,4,5,6,7],220),'unit_price':100.,'promotion_flag':0})

class CommerceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=Commerce(Path(self.tmp.name)/'retail.db')
        self.c.product('o','A','Rice',10000,5000)
        self.c.movement('o','s','A',10,'purchase','start')
        self.c.customer('o','c','Customer')

    def test_sale_is_atomic_idempotent_and_scoped(self):
        self.c.checkout('o','s',{'A':2},customer='c',reference='sale1')
        self.c.checkout('o','s',{'A':2},customer='c',reference='sale1')
        self.assertEqual(self.c.stock('o','s','A')['on_hand'],8)
        self.assertEqual(self.c.rows('loyalty','o')[0]['points'],2)
        self.assertEqual(self.c.stock('other','s','A')['on_hand'],0)
        with self.assertRaises(ValueError):self.c.checkout('o','s',{'A':20})
        self.assertEqual(len(self.c.rows('orders','o','s')),1)
        self.assertEqual(self.c.sales('o','s').sales_qty.sum(),2)

    def test_reservation_cancellation_delivery(self):
        order=self.c.checkout('o','s',{'A':8},source='online',payment='cod',customer='c')
        self.assertEqual(self.c.stock('o','s','A')['available'],2)
        with self.assertRaises(ValueError):self.c.checkout('o','s',{'A':3})
        with self.assertRaises(ValueError):self.c.transition('other','s',order,'accepted')
        for state in ['accepted','picking','packed','ready']:
            self.c.transition('o','s',order,state)
        with self.assertRaises(ValueError):self.c.transition('o','s',order,'dispatched')
        self.c.transition('o','s',order,'dispatched',rider='R1')
        with self.assertRaises(ValueError):self.c.transition('o','s',order,'completed',proof='code')
        self.c.transition('o','s',order,'completed',proof='code',cod_collected=True)
        self.assertEqual(self.c.stock('o','s','A'),{'on_hand':2,'reserved':0,'available':2})
        with self.assertRaises(ValueError):self.c.transition('o','s',order,'completed',proof='code',cod_collected=True)
        second=self.c.checkout('o','s',{'A':2},source='telephone')
        self.c.transition('o','s',second,'cancelled')
        self.assertEqual(self.c.stock('o','s','A')['available'],2)

    def test_ledger_immutable(self):
        import sqlite3
        with self.c.connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM movements')

class AnalyticsTests(unittest.TestCase):
    def test_upload_alias_and_validation(self):
        h=history().rename(columns={'sku':'SKU','unit_price':'price'})
        v=DataValidator()
        self.assertTrue(v.validate_sales_data(h)['is_valid'])
        self.assertIn('unit_price',v.clean_data(h))
        for col,value in [('sales_qty','invalid'),('promotion_flag',2),('date',None),('price',np.inf)]:
            bad=h.copy();bad[col]=bad[col].astype(object);bad.loc[0,col]=value
            self.assertFalse(v.validate_sales_data(bad)['is_valid'])
        self.assertFalse(v.validate_sales_data(h.head(30))['is_valid'])

    def test_forecast_dates_intervals_and_group_guard(self):
        h=history();f,details=forecast_sales(h,14)
        self.assertEqual(len(f),14)
        self.assertEqual(f.date.iloc[0],h.date.max()+pd.Timedelta(days=1))
        self.assertEqual(details['metrics']['mae'],0)
        self.assertTrue((f.lower_bound<=f.forecast).all())
        with self.assertRaises(ValueError):forecast_sales(h,8)
        mixed=pd.concat([h,h.assign(sku='B')])
        with self.assertRaises(ValueError):forecast_sales(mixed)
        with self.assertRaises(ValueError):forecast_sales(h.drop(index=20))

    def test_inventory_and_simulator_conservation(self):
        h=history()
        self.assertTrue(inventory_plan(h,0,50)['reorder_needed'])
        zero=h.assign(sales_qty=0)
        self.assertFalse(inventory_plan(zero,0,50)['reorder_needed'])
        sim=InventorySimulator().compare_policies(h,10)
        for policy in ('baseline','proposed'):
            r=sim[policy]
            self.assertGreaterEqual(r['fill_rate'],0);self.assertLessEqual(r['fill_rate'],100)
            self.assertGreaterEqual(min(r['stock_levels']),0)
            self.assertLessEqual(r['fulfilled_demand'],r['total_demand'])
        # Sufficient stock must not be counted as a shortage after subtraction.
        result=InventorySimulator().simulate_proposed_policy(h.head(1),10,np.array([0]),np.array([0]))
        self.assertEqual(result['stockout_events'],0)

    def test_exports_are_real_files(self):
        table=pd.DataFrame({'sku':['=1+1'],'quantity':[2]})
        csv,_,_=export_table(table)
        self.assertIn(b"'=1+1",csv)
        xlsx,_,_=export_table(table,'Excel');self.assertTrue(xlsx.startswith(b'PK'))

    def test_scoped_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            l=StockLedger(Path(tmp)/'ledger.db')
            l.record('o','s','A',5,'purchase','p1');l.record('o','s','A',5,'purchase','p1')
            self.assertEqual(l.balance('o','s','A'),5)
            self.assertIsNone(l.balance('other','s','A'))
            with self.assertRaises(ValueError):l.record('o','s','A',-6,'sale','s1')

class VisionTests(unittest.TestCase):
    def test_persistence_ack_scope_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.db';store=EventStore(path,'o','s')
            def event(ts):return {'alert_id':uuid4().hex,'timestamp':ts,'camera_id':'C1','alert_type':'loitering','severity_score':3,'confidence':.8,'message':'Review zone','zone_id':None}
            current=event(time.time());store.save(current);store.save(event(time.time()-31*86400))
            reopened=EventStore(path,'o','s')
            self.assertEqual(len(reopened.query(search='review')),1)
            self.assertEqual(EventStore(path,'other','s').query(),[])
            self.assertFalse(EventStore(path,'other','s').acknowledge(current['alert_id']))
            self.assertTrue(reopened.acknowledge(current['alert_id']))
            self.assertEqual(reopened.query(include_acknowledged=False),[])

if __name__=='__main__':unittest.main()
