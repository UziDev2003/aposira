"""Run with: python -m streamlit run aposira/app.py"""
from pathlib import Path
import sys
import json
import runpy
from uuid import uuid4
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'retail-forecasting'))
from aposira.commerce import Commerce

st.set_page_config(page_title='APOSIRA | Retail Intelligence',page_icon='🏪',layout='wide')
commerce=Commerce(ROOT/'data/aposira.db')
st.sidebar.title('APOSIRA')
st.sidebar.caption('One Intelligent System for Modern Retail')
org=st.sidebar.text_input('Organization',value='my-business',key='organization_input').strip()
branch=st.sidebar.text_input('Branch',value='main-store',key='branch_input').strip()
st.sidebar.caption('Local development workspace • organization selection is not authentication')
if not org or not branch:
    st.info('Enter an organization and branch to continue.');st.stop()
page=st.sidebar.radio('Workspace',['Overview','Products & Stock','POS & Orders','Fulfillment & Delivery','Customers & Loyalty','Forecasting & Inventory','Store Vision','Audit Trail'],key='workspace_page')
scope=(org,branch)
if st.session_state.get('aposira_scope') != scope:
    for key in list(st.session_state):
        if key not in ('organization_input','branch_input','workspace_page'):
            del st.session_state[key]
    st.session_state.aposira_scope=scope
st.session_state.aposira_commerce=commerce
st.session_state.aposira_org=org
st.session_state.aposira_branch=branch
st.session_state.aposira_sales=commerce.sales(org,branch)
products=commerce.rows('products',org)
orders=commerce.rows('orders',org,branch)

if page=='Overview':
    st.title('One Intelligent System for Modern Retail')
    st.caption(f'{org} / {branch} • PKR')
    cols=st.columns(4)
    cols[0].metric('Completed sales',f"PKR {sum(o['total'] for o in orders if o['status']=='completed')/100:,.2f}")
    cols[1].metric('Products',len(products))
    cols[2].metric('Open orders',sum(o['status'] not in ('completed','cancelled') for o in orders))
    cols[3].metric('Customers',len(commerce.rows('customers',org)))
    st.markdown('Manage products and stock, take payments at the POS, fulfill orders, and use sales history to plan replenishment. Store Vision provides camera analytics and a retained event timeline.')
    st.info('Start in Products & Stock, add a product and receive stock. Then complete a POS sale or create an order. Import six months of daily sales in Forecasting & Inventory for demand predictions.')
    st.caption('This integrated local prototype does not yet provide production authentication, payment processing, marketplace connectors, or trained shelf-product recognition.')
    if orders:
        st.dataframe(pd.DataFrame(orders).drop(columns=['items']),use_container_width=True)

elif page=='Products & Stock':
    st.title('Products & Stock')
    with st.form('product'):
        sku=st.text_input('SKU');name=st.text_input('Product name')
        price=st.number_input('Selling price (PKR)',min_value=0.,value=100.)
        cost=st.number_input('Unit cost (PKR)',min_value=0.,value=70.)
        if st.form_submit_button('Save product'):
            try:
                commerce.product(org,sku,name,round(price*100),round(cost*100));st.rerun()
            except ValueError as e:st.error(str(e))
    if products:
        st.dataframe(pd.DataFrame([{**p,**commerce.stock(org,branch,p['sku']),'price':p['price']/100,'cost':p['cost']/100} for p in products]),use_container_width=True)
        with st.form('stock'):
            sku=st.selectbox('Product',[p['sku'] for p in products])
            qty=st.number_input('Quantity change (negative for removal)',value=1,step=1)
            reason=st.selectbox('Reason',['purchase','opening_balance','adjustment','return','damage'])
            reference=st.text_input('Unique movement reference')
            if st.form_submit_button('Record stock movement'):
                try:
                    commerce.movement(org,branch,sku,int(qty),reason,reference);st.rerun()
                except ValueError as e:st.error(str(e))
        st.dataframe(commerce.rows('movements',org,branch),use_container_width=True)

elif page=='POS & Orders':
    st.title('POS & Orders')
    if not products:st.info('Add products and receive stock first.');st.stop()
    st.session_state.setdefault('cart_items',{})
    with st.form('cart'):
        sku=st.selectbox('Product',[p['sku'] for p in products],format_func=lambda s:next(f"{p['name']} — PKR {p['price']/100:,.2f}" for p in products if p['sku']==s))
        qty=st.number_input('Quantity',min_value=1,value=1,step=1)
        if st.form_submit_button('Add to cart'):
            st.session_state.cart_items[sku]=st.session_state.cart_items.get(sku,0)+int(qty)
    st.write(st.session_state.cart_items)
    if st.button('Clear cart'):
        st.session_state.cart_items={};st.session_state.pop('checkout_reference',None);st.rerun()
    with st.form('checkout'):
        source=st.selectbox('Order source',['pos','online','telephone'])
        payment=st.selectbox('Payment',['cash','external_paid','cod'])
        customers=commerce.rows('customers',org)
        customer=st.selectbox('Customer (optional)',['']+[c['customer'] for c in customers])
        confirmed=st.checkbox('Payment received (cash / external provider), or cash on delivery agreed')
        st.caption('External paid records a payment already handled by your provider. No card details are collected.')
        if st.form_submit_button('Complete sale / create order'):
            if not confirmed:st.error('Confirm the payment arrangement first.')
            else:
                try:
                    reference=st.session_state.setdefault('checkout_reference',str(uuid4()))
                    order_id=commerce.checkout(org,branch,st.session_state.cart_items,source,customer,payment,reference)
                    st.session_state.cart_items={};st.session_state.pop('checkout_reference',None)
                    st.success(f'Order {order_id} saved')
                    st.session_state.checkout_receipt=next(o for o in commerce.rows('orders',org,branch) if o['id']==order_id)
                except ValueError as e:st.error(str(e))

    if st.session_state.get('checkout_receipt'):
        receipt=st.session_state.checkout_receipt
        st.download_button('Download receipt',json.dumps(receipt,indent=2),file_name=f"aposira-{receipt['id']}.json",mime='application/json')

elif page=='Fulfillment & Delivery':
    st.title('Fulfillment & Delivery')
    active=[o for o in orders if commerce.transitions[o['status']]]
    if not active:st.info('No orders awaiting fulfillment.')
    else:
        selected=st.selectbox('Order',[o['id'] for o in active],format_func=lambda v:next(f"{o['id'][:8]} · {o['status']} · PKR {o['total']/100:,.2f}" for o in active if o['id']==v))
        order=next(o for o in active if o['id']==selected)
        st.dataframe(json.loads(order['items']))
        with st.form('fulfillment'):
            status=st.selectbox('Next status',commerce.transitions[order['status']])
            rider=st.text_input('Assigned rider',value=order['rider'])
            proof=st.text_input('Delivery confirmation code / proof reference')
            collected=st.checkbox('Cash on delivery collected')
            if st.form_submit_button('Update order'):
                try:commerce.transition(org,branch,selected,status,rider,proof,collected);st.rerun()
                except ValueError as e:st.error(str(e))
    if orders:st.dataframe(pd.DataFrame(orders).drop(columns=['items']),use_container_width=True)

elif page=='Customers & Loyalty':
    st.title('Customers & Loyalty')
    with st.form('customer'):
        customer=st.text_input('Customer ID');name=st.text_input('Name')
        if st.form_submit_button('Save customer'):
            try:commerce.customer(org,customer,name);st.rerun()
            except ValueError as e:st.error(str(e))
    points=commerce.rows('loyalty',org)
    st.caption('Earn one point per PKR 100 on completed orders. Redemption and refunds are not yet implemented.')
    st.dataframe([{**c,'points':sum(p['points'] for p in points if p['customer']==c['customer'])} for c in commerce.rows('customers',org)],use_container_width=True)

elif page=='Forecasting & Inventory':
    runpy.run_path(str(ROOT/'retail-forecasting/dashboard/app.py'),init_globals={'APOSIRA_EMBEDDED':True})
elif page=='Store Vision':
    sys.path.insert(0,str(ROOT/'SmartStoreVisionAI'))
    from src.utils.workspace import SCOPE
    scope_token=SCOPE.set((org,branch))
    try:
        runpy.run_path(str(ROOT/'SmartStoreVisionAI/streamlit_app.py'),init_globals={'APOSIRA_EMBEDDED':True})
    except ModuleNotFoundError as e:
        st.error(f'Vision dependency unavailable: {e.name}. Install SmartStoreVisionAI/requirements.txt to run camera inference.')
        from aposira.vision_events import render_timeline
        render_timeline(ROOT,org,branch)
    finally:
        SCOPE.reset(scope_token)
else:
    st.title('Audit Trail')
    rows=commerce.rows('audit',org,branch)
    st.dataframe(rows,use_container_width=True)
    st.download_button('Export audit CSV',pd.DataFrame(rows).to_csv(index=False),'aposira-audit.csv','text/csv')
