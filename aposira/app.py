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
st.markdown("""<style>
.block-container {max-width:1440px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stAppViewContainer"] {background:#f6f8fc;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e2e8f0;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px;}
[data-testid="stForm"] {background:white;border-radius:14px;}
h1,h2,h3 {letter-spacing:-.025em;}
.aposira-hero {background:linear-gradient(115deg,#102c67,#2563eb);color:white;padding:32px;border-radius:20px;margin-bottom:24px;}
.aposira-hero h1 {color:white;font-size:2.2rem;margin:8px 0;}
.aposira-hero p {color:#dbeafe;max-width:680px;margin:0;}
.aposira-eyebrow {font-size:12px;font-weight:700;letter-spacing:.16em;}
</style>""",unsafe_allow_html=True)
commerce=Commerce(ROOT/'data/aposira.db')
st.sidebar.title('APOSIRA')
st.sidebar.caption('One Intelligent System for Modern Retail')
org=st.sidebar.text_input('Organization',value='my-business',key='organization_input').strip()
branch=st.sidebar.text_input('Branch',value='main-store',key='branch_input').strip()
st.sidebar.caption('Demo workspace · PKR')
with st.sidebar.expander('About this workspace'):
    st.caption('Organization and branch select data, not access permissions. This prototype has no login; use demonstration data.')
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
    st.markdown('<div class="aposira-hero"><div class="aposira-eyebrow">APOSIRA · RETAIL INTELLIGENCE</div><h1>Your store, in one place.</h1><p>Track sales, keep stock moving, and turn store activity into better decisions.</p></div>',unsafe_allow_html=True)
    st.caption(f'{org} / {branch} • PKR')
    cols=st.columns(4)
    cols[0].metric('Completed sales',f"PKR {sum(o['total'] for o in orders if o['status']=='completed')/100:,.2f}")
    cols[1].metric('Products',len(products))
    cols[2].metric('Open orders',sum(o['status'] not in ('completed','cancelled') for o in orders))
    cols[3].metric('Customers',len(commerce.rows('customers',org)))
    st.subheader('Your workspaces')
    destinations=[('Products & Stock','Build your catalogue and receive inventory.'),('POS & Orders','Create a cart, record payment, and issue a receipt.'),('Fulfillment & Delivery','Move orders from picking to confirmed delivery.'),('Customers & Loyalty','Manage customers and track earned points.'),('Forecasting & Inventory','Forecast demand and plan replenishment.'),('Store Vision','Review video analytics and retained events.')]
    def open_workspace(destination):
        st.session_state.workspace_page=destination
    for start in (0,3):
        for col,(destination,description) in zip(st.columns(3),destinations[start:start+3]):
            with col.container(border=True):
                st.markdown(f'**{destination}**')
                st.caption(description)
                st.button('Open '+destination,key='open_'+destination,on_click=open_workspace,args=(destination,),width='stretch')
    st.subheader('Recent orders')
    if orders:
        recent=pd.DataFrame(orders).drop(columns=['items']).tail(10).copy()
        recent['total']=recent['total']/100
        st.dataframe(recent.rename(columns={'total':'Total (PKR)'}),width='stretch',hide_index=True)
    else:
        st.info('No orders yet. Add your first product and opening stock, then create a sale in POS & Orders.')

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
        st.dataframe(pd.DataFrame([{**p,**commerce.stock(org,branch,p['sku']),'price':p['price']/100,'cost':p['cost']/100} for p in products]),width='stretch')
        with st.form('stock'):
            sku=st.selectbox('Product',[p['sku'] for p in products])
            qty=st.number_input('Quantity change (negative for removal)',value=1,step=1)
            reason=st.selectbox('Reason',['purchase','opening_balance','adjustment','return','damage'])
            reference=st.text_input('Unique movement reference')
            if st.form_submit_button('Record stock movement'):
                try:
                    commerce.movement(org,branch,sku,int(qty),reason,reference);st.rerun()
                except ValueError as e:st.error(str(e))
        st.dataframe(commerce.rows('movements',org,branch),width='stretch')

elif page=='POS & Orders':
    st.title('POS & Orders')
    if not products:st.info('Add products and receive stock first.');st.stop()
    st.session_state.setdefault('cart_items',{})
    with st.form('cart'):
        sku=st.selectbox('Product',[p['sku'] for p in products],format_func=lambda s:next(f"{p['name']} — PKR {p['price']/100:,.2f}" for p in products if p['sku']==s))
        qty=st.number_input('Quantity',min_value=1,value=1,step=1)
        if st.form_submit_button('Add to cart'):
            st.session_state.cart_items[sku]=st.session_state.cart_items.get(sku,0)+int(qty)
    cart_rows=[]
    for cart_sku,quantity in st.session_state.cart_items.items():
        product=next(p for p in products if p['sku']==cart_sku)
        cart_rows.append({'Product':product['name'],'SKU':cart_sku,'Quantity':quantity,'Unit price (PKR)':product['price']/100,'Total (PKR)':product['price']*quantity/100})
    if cart_rows:
        st.dataframe(cart_rows,width='stretch',hide_index=True)
        st.metric('Cart total',f"PKR {sum(row['Total (PKR)'] for row in cart_rows):,.2f}")
    else:
        st.info('Your cart is empty. Choose a product and quantity above.')
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
    if orders:st.dataframe(pd.DataFrame(orders).drop(columns=['items']),width='stretch')

elif page=='Customers & Loyalty':
    st.title('Customers & Loyalty')
    with st.form('customer'):
        customer=st.text_input('Customer ID');name=st.text_input('Name')
        if st.form_submit_button('Save customer'):
            try:commerce.customer(org,customer,name);st.rerun()
            except ValueError as e:st.error(str(e))
    points=commerce.rows('loyalty',org)
    st.caption('Earn one point per PKR 100 on completed orders. Redemption and refunds are not yet implemented.')
    st.dataframe([{**c,'points':sum(p['points'] for p in points if p['customer']==c['customer'])} for c in commerce.rows('customers',org)],width='stretch')

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
    st.dataframe(rows,width='stretch')
    st.download_button('Export audit CSV',pd.DataFrame(rows).to_csv(index=False),'aposira-audit.csv','text/csv')
