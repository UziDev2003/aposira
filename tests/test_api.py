import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
from backend import main
from backend.database import Database


def test_upload_forecast_export_and_errors():
    with tempfile.TemporaryDirectory() as tmp:
        original=main.db
        main.db=Database(Path(tmp)/'api.db')
        try:
            client=TestClient(main.app)
            assert client.get('/health').status_code==200
            assert client.post('/api/v1/forecast/generate',json={'sku':'x','store_id':'s'}).status_code==404
            assert client.post('/api/v1/forecast/generate',json={'sku':'x','store_id':'s','horizon':8}).status_code==422
            assert client.post('/api/v1/upload/sales',files={'file':('data.txt',b'bad')}).status_code==400
            h=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=220),'sku':'A','store_id':'s','sales_qty':10,'price':50,'promotion_flag':0})
            assert client.post('/api/v1/upload/sales',files={'file':('sales.csv',h.to_csv(index=False).encode())}).status_code==200
            result=client.post('/api/v1/forecast/generate',json={'sku':'A','store_id':'s'})
            assert result.status_code==200
            assert len(result.json()['forecast'])==7
            assert client.get('/api/v1/forecast/results',params={'sku':'A','store_id':'s'}).json()['count']==7
            assert client.get('/api/v1/forecast/results',params={'sku':"A' OR 1=1 --",'store_id':'s'}).status_code==404
            for fmt,signature in [('CSV',b'\xef\xbb\xbf'),('Excel',b'PK'),('PDF',b'%PDF')]:
                exported=client.get('/api/v1/reports/export',params={'sku':'A','store_id':'s','format_type':fmt})
                assert exported.status_code==200
                assert exported.content.startswith(signature)
        finally:
            main.db.close();main.db=original
