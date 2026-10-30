"""APOSIRA forecasting workspace, standalone or embedded in the unified app."""
from pathlib import Path
import sys
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.core.data_validator import DataValidator
from backend.core.decision_service import forecast_sales, inventory_plan, daily_history
from backend.core.optimization.abc_analysis import calculate_abc_classification
from backend.core.optimization.simulator import InventorySimulator
from backend.core.stock_ledger import StockLedger

if not globals().get('APOSIRA_EMBEDDED'):
    st.set_page_config(page_title='APOSIRA | Forecasting',page_icon='📊',layout='wide')
st.title('AI Analytics & Forecasting')
st.caption('APOSIRA • Predictive Demand Forecasting & Intelligent Inventory Optimization • PKR')
embedded=bool(globals().get('APOSIRA_EMBEDDED'))
org=st.session_state.get('aposira_org','local')
branch=st.session_state.get('aposira_branch','')
commerce=st.session_state.get('aposira_commerce') if embedded else None
source=st.radio('Sales source',['POS sales','Uploaded history','Sample demonstration'] if embedded else ['Uploaded history','Sample demonstration'],horizontal=True,key='retail_source')
if source=='Sample demonstration':
    st.warning('Demonstration data: these are sample transactions, not your live store sales.')
    sales=pd.read_csv(ROOT/'data/sample/sample_sales.csv')
    products=pd.read_csv(ROOT/'data/sample/sample_products.csv')
elif source=='Uploaded history':
    upload=st.file_uploader('Sales CSV or JSON',type=['csv','json'],key='retail_upload')
    st.caption('Required: date, sku (or SKU), store_id, sales_qty, price (or unit_price), promotion_flag (0/1). At least six months per SKU/store. Daily gaps must be supplied explicitly before forecasting.')
    if upload is not None and st.button('Validate and use history',key='retail_validate'):
        try:
            df=pd.read_csv(upload) if upload.name.lower().endswith('.csv') else pd.read_json(upload)
            validator=DataValidator();result=validator.validate_sales_data(df)
            if result['is_valid']:
                st.session_state.retail_history=validator.clean_data(df)
                st.session_state.pop('retail_forecast',None)
                st.success(f'Validated {len(df):,} records')
            else:
                for error in result['errors']:st.error(error)
        except (ValueError,TypeError) as e:st.error(f'Could not read sales data: {e}')
    sales=st.session_state.get('retail_history',pd.DataFrame())
    products=pd.DataFrame()
else:
    sales=st.session_state.get('aposira_sales',pd.DataFrame())
    products=pd.DataFrame()
if sales.empty:
    st.info('No sales records available. Complete a POS sale or import historical data. Forecasting requires six months of history.');st.stop()
sales=sales.copy();sales['date']=pd.to_datetime(sales['date'])
# Imported histories in the unified app are restricted to the selected branch.
if embedded and source!='Sample demonstration':
    sales=sales[sales.store_id.astype(str)==branch]
    if sales.empty:st.info('No records match the selected branch.');st.stop()
store=st.selectbox('Store',sorted(sales.store_id.astype(str).unique()),key='retail_store')
filtered=sales[sales.store_id.astype(str)==store]
sku=st.selectbox('SKU',sorted(filtered.sku.astype(str).unique()),key='retail_sku')
history=filtered[filtered.sku.astype(str)==sku]
cols=st.columns(3)
cols[0].metric('Revenue',f"PKR {(filtered.sales_qty*filtered.unit_price).sum():,.2f}")
cols[1].metric('Active SKUs',filtered.sku.nunique())
cols[2].metric('History days',(history.date.max()-history.date.min()).days+1)
st.line_chart(history.groupby('date').sales_qty.sum())
section=st.radio('Analysis',['Demand Forecast','Inventory & Alerts','ABC Analysis','Policy Simulator','Reports'],horizontal=True,key='retail_section')
context=(org,store,sku,source,int(pd.util.hash_pandas_object(history,index=False).sum()))
if section=='Demand Forecast':
    model=st.selectbox('Model',['seasonal_naive','lightgbm','prophet','ensemble'],key='retail_model')
    horizon=st.selectbox('Horizon (days)',[7,14,30],key='retail_horizon')
    st.caption('Seasonal naive is a weekly baseline. Ensemble combines LightGBM and Prophet; ARIMA/LSTM are not implemented. Accuracy uses a separate final 30-day holdout.')
    if st.button('Generate forecast',key='retail_generate'):
        try:
            with st.spinner('Evaluating model and forecasting…'):
                forecast,details=forecast_sales(history,horizon,model)
            st.session_state.retail_forecast=(context,forecast,details)
        except (ValueError,ImportError,RuntimeError,OSError) as e:st.error(str(e))
    saved=st.session_state.get('retail_forecast')
    if saved and saved[0]==context:
        _,forecast,details=saved
        fig=go.Figure()
        fig.add_scatter(x=history.date,y=history.sales_qty,name='Actual')
        fig.add_scatter(x=forecast.date,y=forecast.upper_bound,line={'width':0},name='Upper bound')
        fig.add_scatter(x=forecast.date,y=forecast.lower_bound,fill='tonexty',line={'width':0},name='Nominal 95% interval')
        fig.add_scatter(x=forecast.date,y=forecast.forecast,name='Forecast')
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(forecast,use_container_width=True)
        st.json(details)
        if details['retraining_recommended']:st.warning('MAPE is unavailable or above 15%. Review the data and model before ordering.')
        st.download_button('Download forecast CSV',forecast.to_csv(index=False),'aposira-forecast.csv','text/csv')

elif section=='Inventory & Alerts':
    ledger=StockLedger(ROOT/'data/stock_ledger.db')
    live=commerce is not None and source!='Sample demonstration'
    current=commerce.stock(org,store,sku)['available'] if live else ledger.balance(org,store,sku)
    if not live:
        with st.form('retail_movement'):
            change=st.number_input('Stock movement (+ received / − removed)',value=1.,step=1.)
            reason=st.text_input('Reason');reference=st.text_input('Unique reference')
            if st.form_submit_button('Record movement'):
                try:ledger.record(org,store,sku,change,reason,reference);st.rerun()
                except ValueError as e:st.error(str(e))
    if current is None:st.info('Record opening stock to generate inventory alerts.');st.stop()
    st.metric('Available stock',current)
    c1,c2,c3=st.columns(3)
    cost=c1.number_input('Unit cost (PKR)',min_value=.01,value=100.)
    lead=c2.number_input('Supplier lead time (days)',min_value=0,value=5)
    service=c3.slider('Service level',.50,.99,.95)
    order_cost=st.number_input('Cost per order (PKR)',min_value=0.,value=500.)
    holding=st.number_input('Annual holding cost fraction',min_value=.001,value=.20)
    try:
        plan=inventory_plan(history,current,cost,lead,order_cost,holding,service)
        st.session_state.retail_plan=(context,pd.DataFrame([{'sku':sku,'store_id':store,**plan}]))
        if plan['status']!='Normal':st.warning(f"{plan['status']}: review replenishment. Suggested quantity: {plan['recommended_quantity']:,.0f}")
        else:st.success('Stock is above the reorder point and seven-day supply threshold.')
        st.dataframe(st.session_state.retail_plan[1],use_container_width=True)
        st.caption('Recommendations use observed daily demand and your cost assumptions. They do not submit purchase orders.')
    except ValueError as e:st.error(str(e))

elif section=='ABC Analysis':
    abc=calculate_abc_classification(filtered,products)
    st.dataframe(abc,use_container_width=True)
    st.caption('Value thresholds are 80% / 95%; item counts are measured, not fixed to 20% / 30% / 50%.')
    st.download_button('Export ABC CSV',abc.to_csv(index=False),'aposira-abc.csv','text/csv')

elif section=='Policy Simulator':
    initial=st.number_input('Initial stock',min_value=0.,value=1000.)
    if st.button('Compare policies'):
        try:
            result=InventorySimulator().compare_policies(daily_history(history),initial)
            table=pd.DataFrame({k:{m:v for m,v in result[k].items() if m!='stock_levels'} for k in ('baseline','proposed')})
            st.session_state.retail_simulation=(context,table.reset_index(names='metric'))
            st.dataframe(table,use_container_width=True)
            st.line_chart(pd.DataFrame({k:result[k]['stock_levels'] for k in ('baseline','proposed')}))
            st.caption('Historical simulation with five-day lead time. Proposed uses prior-demand averages and safety stock; savings are not guaranteed.')
        except ValueError as e:st.error(str(e))

else:
    from backend.core.reports import export_table
    name=st.selectbox('Report',['Sales','Forecast','Inventory plan','ABC analysis','Simulation'])
    start=st.date_input('From',value=history.date.min().date())
    end=st.date_input('Through',value=history.date.max().date())
    table=None
    if name=='Sales':table=history[(history.date.dt.date>=start)&(history.date.dt.date<=end)]
    elif name=='ABC analysis':table=calculate_abc_classification(filtered[(filtered.date.dt.date>=start)&(filtered.date.dt.date<=end)],products)
    else:
        key={'Forecast':'retail_forecast','Inventory plan':'retail_plan','Simulation':'retail_simulation'}[name]
        saved=st.session_state.get(key)
        if saved and saved[0]==context:table=saved[1]
    fmt=st.selectbox('Format',['CSV','Excel','PDF'])
    if table is None:st.info('Generate this analysis for the current SKU/store first.')
    elif start>end:st.error('Start date must be before end date.')
    elif st.button('Prepare download'):
        try:
            data,mime,extension=export_table(table,fmt,f'APOSIRA {name}')
            st.download_button('Download report',data,f'aposira-report.{extension}',mime)
        except ImportError as e:st.error(f'Install export dependency: {e}')
