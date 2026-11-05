"""APOSIRA Unified Retail Intelligence Platform.
Run with: python -m streamlit run aposira/app.py
"""
from pathlib import Path
import sys
import json
import runpy
from uuid import uuid4
from datetime import datetime
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / 'aposira'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from aposira.commerce import Commerce
from aposira.ui import apply_theme, page_header

st.set_page_config(
    page_title='APOSIRA | Retail Intelligence',
    page_icon='🏪',
    layout='wide',
    initial_sidebar_state='expanded'
)

apply_theme()

commerce = Commerce(ROOT / 'data/aposira.db')

# Sidebar Navigation & Scope
st.sidebar.title('APOSIRA')
st.sidebar.caption('Intelligent Retail Operating System')

st.sidebar.markdown("---")
org = st.sidebar.text_input('Organization', value='my-business', key='organization_input').strip()
branch = st.sidebar.text_input('Branch', value='main-store', key='branch_input').strip()
st.sidebar.caption('Currency: **PKR** • Scoped Ledger')

with st.sidebar.expander('Workspace Information'):
    st.caption('Organization and branch partition transactional state, vision events, and stock movements. Demonstration mode has no external authentication.')

if not org or not branch:
    st.info('Enter an organization and branch to continue.')
    st.stop()

page = st.sidebar.radio(
    'Workspace',
    [
        'Overview',
        'Products & Stock',
        'POS & Orders',
        'Fulfillment & Delivery',
        'Customers & Loyalty',
        'Forecasting & Inventory',
        'Store Vision',
        'Audit Trail'
    ],
    key='workspace_page'
)

# Maintain strict branch/org isolation
scope = (org, branch)
if st.session_state.get('aposira_scope') != scope:
    for key in list(st.session_state):
        if key not in ('organization_input', 'branch_input', 'workspace_page'):
            del st.session_state[key]
    st.session_state.aposira_scope = scope

st.session_state.aposira_commerce = commerce
st.session_state.aposira_org = org
st.session_state.aposira_branch = branch
st.session_state.aposira_sales = commerce.sales(org, branch)

products = commerce.rows('products', org)
orders = commerce.rows('orders', org, branch)


# -----------------------------------------------------------------------------
# OVERVIEW PAGE
# -----------------------------------------------------------------------------
if page == 'Overview':
    st.markdown(
        f'<div class="aposira-hero">'
        f'<div class="aposira-eyebrow">APOSIRA · RETAIL INTELLIGENCE</div>'
        f'<h1>Your store, in one place.</h1>'
        f'<p>Real-time point of sale, immutable inventory ledger, predictive demand forecasting, and computer-vision store surveillance.</p>'
        f'<div style="margin-top: 14px; font-size: 0.85rem; color: #bfdbfe;">Active Scope: <b>{org}</b> / <b>{branch}</b> • Currency: <b>PKR</b></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    completed_orders = [o for o in orders if o['status'] == 'completed']
    gross_revenue = sum(o['total'] for o in completed_orders) / 100
    open_orders_count = sum(o['status'] not in ('completed', 'cancelled') for o in orders)
    customers_count = len(commerce.rows('customers', org))

    # Calculate inventory on hand across products
    total_stock_units = 0
    low_stock_items = 0
    for p in products:
        s = commerce.stock(org, branch, p['sku'])
        total_stock_units += s['on_hand']
        if s['available'] <= 5:
            low_stock_items += 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Completed Sales', f"PKR {gross_revenue:,.2f}")
    col2.metric('Active Catalogue', f"{len(products)} SKUs", f"{total_stock_units:,} units in stock")
    col3.metric('Open Orders', f"{open_orders_count}", "Awaiting fulfillment" if open_orders_count else "All caught up")
    col4.metric('Registered Customers', f"{customers_count}")

    if low_stock_items > 0:
        st.warning(f"⚠️ **Attention Required**: {low_stock_items} product(s) in branch **{branch}** have low or zero stock. Review **Products & Stock** to reorder.")

    st.markdown("### Workspaces & Workflows")
    destinations = [
        ('Products & Stock', 'Build your catalogue, monitor stock levels, and record immutable movements.'),
        ('POS & Orders', 'Fast POS terminal, itemized cart, payment confirmation, and receipt issuing.'),
        ('Fulfillment & Delivery', 'Track order states: Picking, Packing, Ready, Rider Dispatch, and Proof of Delivery.'),
        ('Customers & Loyalty', 'Manage customer profiles and automatic loyalty points accrual (1 pt / PKR 100).'),
        ('Forecasting & Inventory', 'Predictive demand forecasting (LightGBM/Prophet), EOQ, safety stock, and ABC analysis.'),
        ('Store Vision', 'AI computer vision pipeline, heatmaps, dwell times, and 30-day security event timeline.')
    ]

    def open_workspace(destination):
        st.session_state.workspace_page = destination

    for start in (0, 3):
        for col, (destination, description) in zip(st.columns(3), destinations[start:start+3]):
            with col.container(border=True):
                st.markdown(f"**{destination}**")
                st.caption(description)
                st.button('Open ' + destination, key='open_' + destination, on_click=open_workspace, args=(destination,), width='stretch')

    st.markdown("### Recent Branch Orders")
    if orders:
        recent_df = pd.DataFrame(orders).drop(columns=['items']).tail(10).copy()
        recent_df['Total (PKR)'] = recent_df['total'].apply(lambda x: f"PKR {x/100:,.2f}")
        recent_df = recent_df.drop(columns=['total', 'org', 'branch'], errors='ignore')
        recent_df = recent_df.rename(columns={
            'id': 'Order ID',
            'source': 'Source',
            'customer': 'Customer ID',
            'payment': 'Payment Method',
            'status': 'Status',
            'rider': 'Assigned Rider',
            'proof': 'Delivery Proof',
            'created_at': 'Created At'
        })
        st.dataframe(recent_df, width='stretch', hide_index=True)
    else:
        st.info('No orders recorded yet. Add products with opening stock, then process sales in POS & Orders.')


# -----------------------------------------------------------------------------
# PRODUCTS & STOCK PAGE
# -----------------------------------------------------------------------------
elif page == 'Products & Stock':
    page_header('Products & Stock', 'Manage catalogue items and record permanent, immutable stock movements.', 'APOSIRA · INVENTORY')

    tab_catalogue, tab_movement, tab_new = st.tabs(['Catalogue & Stock Health', 'Record Stock Movement', 'Add New Product'])

    with tab_new:
        with st.form('product'):
            st.subheader('Add Product to Catalogue')
            pcol1, pcol2 = st.columns(2)
            sku = pcol1.text_input('Product SKU (e.g., SKU001)')
            name = pcol2.text_input('Product Name')
            price = pcol1.number_input('Selling Price (PKR)', min_value=0.0, value=100.0, step=10.0)
            cost = pcol2.number_input('Unit Cost (PKR)', min_value=0.0, value=70.0, step=10.0)
            
            margin = price - cost
            margin_pct = (margin / price * 100) if price > 0 else 0
            st.caption(f"Estimated Gross Margin: **PKR {margin:,.2f}** ({margin_pct:.1f}%)")

            if st.form_submit_button('Save product'):
                if not sku.strip() or not name.strip():
                    st.error('Both Product SKU and Name are required.')
                else:
                    try:
                        commerce.product(org, sku.strip(), name.strip(), round(price * 100), round(cost * 100))
                        st.success(f"Product '{name}' ({sku}) saved successfully.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    with tab_catalogue:
        if products:
            catalog_rows = []
            for p in products:
                stk = commerce.stock(org, branch, p['sku'])
                status = "In Stock" if stk['available'] > 5 else ("Low Stock" if stk['available'] > 0 else "Out of Stock")
                catalog_rows.append({
                    'SKU': p['sku'],
                    'Product Name': p['name'],
                    'Selling Price (PKR)': p['price'] / 100,
                    'Unit Cost (PKR)': p['cost'] / 100,
                    'Gross Margin (%)': round((p['price'] - p['cost']) / p['price'] * 100, 1) if p['price'] > 0 else 0.0,
                    'On Hand': stk['on_hand'],
                    'Reserved': stk['reserved'],
                    'Available': stk['available'],
                    'Status': status
                })

            st.dataframe(pd.DataFrame(catalog_rows), width='stretch', hide_index=True)
        else:
            st.info('No products in catalogue yet. Use the "Add New Product" tab above.')

    with tab_movement:
        if not products:
            st.info('Please create at least one product before recording stock movements.')
        else:
            with st.form('stock'):
                st.subheader('Record Stock Movement (Purchase, Return, Adjustment)')
                mcol1, mcol2 = st.columns(2)
                sku_choice = mcol1.selectbox('Select Product', [p['sku'] for p in products], format_func=lambda s: next(f"{p['name']} ({s})" for p in products if p['sku'] == s))
                qty = mcol2.number_input('Quantity Change (positive for in, negative for out)', value=10, step=1)
                reason = mcol1.selectbox('Movement Reason', ['purchase', 'opening_balance', 'adjustment', 'return', 'damage'])
                ref_default = f"MOV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                reference = mcol2.text_input('Movement Reference', value=ref_default)

                if st.form_submit_button('Record stock movement'):
                    try:
                        commerce.movement(org, branch, sku_choice, int(qty), reason, reference.strip())
                        st.success(f"Stock movement recorded: {qty:+d} units for SKU {sku_choice}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

            st.markdown("#### Movement History")
            movements = commerce.rows('movements', org, branch)
            if movements:
                m_df = pd.DataFrame(movements).drop(columns=['org', 'branch'], errors='ignore')
                st.dataframe(m_df.rename(columns={
                    'id': 'ID', 'sku': 'SKU', 'quantity': 'Quantity',
                    'reason': 'Reason', 'reference': 'Reference', 'created_at': 'Timestamp'
                }), width='stretch', hide_index=True)
            else:
                st.info('No stock movements recorded for this branch.')


# -----------------------------------------------------------------------------
# POS & ORDERS PAGE
# -----------------------------------------------------------------------------
elif page == 'POS & Orders':
    page_header('POS & Orders', 'Create sales orders, confirm payments, deduct stock atomically, and generate receipts.', 'APOSIRA · POINT OF SALE')

    if not products:
        st.info('Add products and receive stock first.')
        st.stop()

    st.session_state.setdefault('cart_items', {})

    col_cart_build, col_checkout = st.columns([1, 1], gap='large')

    with col_cart_build:
        st.subheader('1. Cart Builder')

        # Labels must stay stable across reruns (live stock in the label changes the widget
        # identity after every sale), so availability is shown separately.
        product_labels = {p['sku']: f"{p['name']} ({p['sku']}) — PKR {p['price']/100:,.2f}" for p in products}
        stock_by_sku = {p['sku']: commerce.stock(org, branch, p['sku']) for p in products}

        with st.form('cart'):
            sku = st.selectbox('Select Product', list(product_labels), format_func=product_labels.get)
            qty = st.number_input('Quantity', min_value=1, value=1, step=1)

            if st.form_submit_button('Add to cart'):
                available = stock_by_sku[sku]['available']
                requested = st.session_state.cart_items.get(sku, 0) + int(qty)
                if requested > available:
                    st.error(f"Only {available} unit(s) of {sku} available; the cart would hold {requested}.")
                else:
                    st.session_state.cart_items[sku] = requested
        st.caption('Available stock: ' + ' · '.join(f"{s}: {v['available']}" for s, v in stock_by_sku.items()))

        cart_rows = []
        for cart_sku, quantity in st.session_state.cart_items.items():
            product = next(p for p in products if p['sku'] == cart_sku)
            cart_rows.append({
                'Product': product['name'],
                'SKU': cart_sku,
                'Quantity': quantity,
                'Unit price (PKR)': product['price'] / 100,
                'Total (PKR)': product['price'] * quantity / 100
            })

        if cart_rows:
            st.dataframe(cart_rows, width='stretch', hide_index=True)
            cart_total = sum(row['Total (PKR)'] for row in cart_rows)
            st.metric('Cart total', f"PKR {cart_total:,.2f}")
        else:
            st.info('Your cart is empty. Choose a product and quantity above.')

        if st.button('Clear cart'):
            st.session_state.cart_items = {}
            st.session_state.pop('checkout_reference', None)
            st.rerun()

    with col_checkout:
        st.subheader('2. Checkout & Payment')
        with st.form('checkout'):
            source = st.selectbox('Order source', ['pos', 'online', 'telephone'])
            payment = st.selectbox('Payment', ['cash', 'external_paid', 'cod'])
            customers = commerce.rows('customers', org)
            customer = st.selectbox('Customer (optional)', [''] + [c['customer'] for c in customers])
            # CRITICAL: This confirmed checkbox MUST remain the FIRST checkbox on this page for test_ui.py
            confirmed = st.checkbox('Payment received (cash / external provider), or cash on delivery agreed')
            st.caption('External paid records a payment handled by an integrated terminal/provider. No raw card numbers stored.')
            
            if st.form_submit_button('Complete sale / create order'):
                if not confirmed:
                    st.error('Confirm the payment arrangement first.')
                elif not st.session_state.cart_items:
                    st.error('Your cart is empty.')
                else:
                    try:
                        reference = st.session_state.setdefault('checkout_reference', str(uuid4()))
                        order_id = commerce.checkout(
                            org, branch, st.session_state.cart_items, source, customer, payment, reference
                        )
                        st.session_state.cart_items = {}
                        st.session_state.pop('checkout_reference', None)
                        st.success(f'Order {order_id} saved successfully')
                        st.session_state.checkout_receipt = next(
                            o for o in commerce.rows('orders', org, branch) if o['id'] == order_id
                        )
                    except ValueError as e:
                        st.error(str(e))

        if st.session_state.get('checkout_receipt'):
            receipt = st.session_state.checkout_receipt
            st.markdown(
                f'<div class="receipt-box">'
                f'<div style="text-align:center;font-weight:700;font-size:1.1rem;color:#0f172a;">APOSIRA RECEIPT</div>'
                f'<div style="text-align:center;font-size:0.85rem;color:#64748b;">Branch: {branch} • {receipt.get("created_at", "")}</div>'
                f'<hr style="margin:12px 0;border-color:#e2e8f0;"/>'
                f'<div style="font-size:0.85rem;"><b>Order ID:</b> {receipt["id"]}</div>'
                f'<div style="font-size:0.85rem;"><b>Status:</b> <span class="status-badge badge-{receipt["status"]}">{receipt["status"]}</span></div>'
                f'<div style="font-size:0.85rem;"><b>Payment:</b> {receipt["payment"].upper()} ({receipt["source"].upper()})</div>'
                f'<div style="font-size:0.85rem;"><b>Customer:</b> {receipt["customer"] or "Walk-in Customer"}</div>'
                f'<div style="text-align:right;font-size:1.2rem;font-weight:700;color:#1e3a8a;margin-top:8px;">Total: PKR {receipt["total"]/100:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.download_button(
                'Download receipt',
                json.dumps(receipt, indent=2),
                file_name=f"aposira-{receipt['id']}.json",
                mime='application/json'
            )


# -----------------------------------------------------------------------------
# FULFILLMENT & DELIVERY PAGE
# -----------------------------------------------------------------------------
elif page == 'Fulfillment & Delivery':
    page_header('Fulfillment & Delivery', 'Progress orders: picking → packing → ready → dispatch → delivery.', 'APOSIRA · FULFILLMENT')

    active = [o for o in orders if commerce.transitions.get(o['status'])]
    if not active:
        st.info('No orders awaiting fulfillment.')
    else:
        selected = st.selectbox(
            'Order',
            [o['id'] for o in active],
            format_func=lambda v: next(f"{o['id'][:8]} · Status: {o['status'].upper()} · PKR {o['total']/100:,.2f} ({o['source']})" for o in active if o['id'] == v)
        )
        order = next(o for o in active if o['id'] == selected)

        fcol1, fcol2 = st.columns([1, 1], gap='large')
        with fcol1:
            st.subheader('Order Details')
            st.markdown(f"**Order ID:** `{order['id']}`")
            st.markdown(f"**Current Status:** <span class='status-badge badge-{order['status']}'>{order['status']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Payment Method:** {order['payment'].upper()}")
            st.markdown(f"**Order Total:** PKR {order['total']/100:,.2f}")
            st.caption("Ordered Items:")
            st.dataframe(pd.DataFrame(json.loads(order['items'])).rename(columns={
                'sku': 'SKU', 'name': 'Item', 'quantity': 'Qty', 'price': 'Unit Paisa'
            }), width='stretch', hide_index=True)

        with fcol2:
            st.subheader('Update Order Status')
            with st.form('fulfillment'):
                status = st.selectbox('Next status', commerce.transitions[order['status']])
                rider = st.text_input('Assigned rider', value=order['rider'])
                proof = st.text_input('Delivery confirmation code / proof reference')
                collected = st.checkbox('Cash on delivery collected')

                if status == 'dispatched':
                    st.caption('ℹ️ A rider name must be provided before dispatch.')
                elif status == 'completed' and order['payment'] == 'cod':
                    st.caption('ℹ️ Cash on delivery collection must be checked to complete order.')

                if st.form_submit_button('Update order'):
                    try:
                        commerce.transition(org, branch, selected, status, rider, proof, collected)
                        st.success(f"Order updated to '{status}'")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    st.markdown("### All Branch Orders")
    if orders:
        orders_df = pd.DataFrame(orders).drop(columns=['items', 'org', 'branch'], errors='ignore').copy()
        orders_df['total'] = orders_df['total'] / 100
        st.dataframe(orders_df.rename(columns={
            'id': 'Order ID', 'source': 'Source', 'customer': 'Customer',
            'total': 'Total (PKR)', 'payment': 'Payment', 'status': 'Status',
            'rider': 'Rider', 'proof': 'Proof', 'created_at': 'Timestamp'
        }), width='stretch', hide_index=True)


# -----------------------------------------------------------------------------
# CUSTOMERS & LOYALTY PAGE
# -----------------------------------------------------------------------------
elif page == 'Customers & Loyalty':
    page_header('Customers & Loyalty', 'Customers earn 1 loyalty point per PKR 100 spent on completed orders.', 'APOSIRA · CUSTOMERS')

    cust_col1, cust_col2 = st.columns([1, 2], gap='large')
    with cust_col1:
        st.subheader('Register Customer')
        with st.form('customer'):
            customer = st.text_input('Customer ID (Phone or Code)')
            name = st.text_input('Full Name')
            if st.form_submit_button('Save customer'):
                if not customer.strip() or not name.strip():
                    st.error('Both Customer ID and Name are required.')
                else:
                    try:
                        commerce.customer(org, customer.strip(), name.strip())
                        st.success(f"Customer '{name}' saved.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    with cust_col2:
        st.subheader('Customer Accounts & Points')
        points = commerce.rows('loyalty', org)
        cust_list = commerce.rows('customers', org)

        if cust_list:
            display_rows = []
            for c in cust_list:
                pts = sum(p['points'] for p in points if p['customer'] == c['customer'])
                tier = "🥇 Gold" if pts >= 200 else ("🥈 Silver" if pts >= 50 else ("🥉 Bronze" if pts > 0 else "🌱 New"))
                display_rows.append({
                    'Customer ID': c['customer'],
                    'Name': c['name'],
                    'Points': pts,
                    'Tier': tier
                })
            st.dataframe(pd.DataFrame(display_rows), width='stretch', hide_index=True)
        else:
            st.info('No customers registered yet.')


# -----------------------------------------------------------------------------
# FORECASTING & INVENTORY EMBEDDED
# -----------------------------------------------------------------------------
elif page == 'Forecasting & Inventory':
    runpy.run_path(str(APP_DIR / 'forecasting/dashboard/app.py'), init_globals={'APOSIRA_EMBEDDED': True})


# -----------------------------------------------------------------------------
# STORE VISION EMBEDDED
# -----------------------------------------------------------------------------
elif page == 'Store Vision':
    from aposira.vision.src.utils.workspace import SCOPE
    scope_token = SCOPE.set((org, branch))
    try:
        runpy.run_path(str(APP_DIR / 'vision/streamlit_app.py'), init_globals={'APOSIRA_EMBEDDED': True})
    except (ModuleNotFoundError, ImportError) as e:
        st.warning(f'Store Vision live camera inference requires optional ML dependencies: {e}. Showing historical security timeline.')
        from aposira.vision_events import render_timeline
        render_timeline(org, branch)
    finally:
        SCOPE.reset(scope_token)


# -----------------------------------------------------------------------------
# AUDIT TRAIL PAGE
# -----------------------------------------------------------------------------
else:
    page_header('Audit Trail', 'Immutable log of order creation and fulfillment transitions for this branch.', 'APOSIRA · AUDIT')

    rows = commerce.rows('audit', org, branch)
    if rows:
        audit_df = pd.DataFrame(rows).drop(columns=['org', 'branch'], errors='ignore')
        st.dataframe(audit_df.rename(columns={
            'id': 'Log ID', 'action': 'Action', 'reference': 'Reference ID',
            'detail': 'Details / Status', 'created_at': 'Timestamp'
        }), width='stretch', hide_index=True)
        st.download_button('Export audit CSV', pd.DataFrame(rows).to_csv(index=False), 'aposira-audit.csv', 'text/csv')
    else:
        st.info('No audit events recorded for this branch yet.')
        st.download_button('Export audit CSV', pd.DataFrame().to_csv(index=False), 'aposira-audit.csv', 'text/csv')
