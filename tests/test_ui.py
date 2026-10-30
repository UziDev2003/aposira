import tempfile
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from aposira.commerce import Commerce


def test_navigation_and_checkout():
    with tempfile.TemporaryDirectory() as tmp:
        db=Commerce(Path(tmp)/'ui.db')
        db.product('my-business','A','Rice',10000,5000)
        db.movement('my-business','main-store','A',20,'opening','open')
        with patch('aposira.commerce.Commerce',return_value=db):
            app=AppTest.from_file(Path(__file__).resolve().parents[1]/'aposira/app.py',default_timeout=30).run()
            assert not app.exception
            for page in ['Products & Stock','Customers & Loyalty','Fulfillment & Delivery','Audit Trail','Forecasting & Inventory','POS & Orders']:
                app.radio(key='workspace_page').set_value(page).run()
                assert not app.exception, (page,list(app.exception))
            next(b for b in app.button if b.label=='Add to cart').click().run()
            app.checkbox[0].check()
            next(b for b in app.button if b.label=='Complete sale / create order').click().run()
            assert not app.exception
            assert db.stock('my-business','main-store','A')['on_hand']==19
            assert len(db.sales('my-business','main-store'))==1
            app.radio(key='workspace_page').set_value('Forecasting & Inventory').run()
            assert not app.exception
            app.radio(key='retail_source').set_value('Sample demonstration').run()
            assert not app.exception
            next(b for b in app.button if b.label=='Generate forecast').click().run()
            assert not app.exception
            assert 'retail_forecast' in app.session_state
            app.radio(key='retail_section').set_value('Policy Simulator').run()
            next(b for b in app.button if b.label=='Compare policies').click().run()
            assert not app.exception


def test_branch_change_clears_prior_camera_and_cart_state():
    with tempfile.TemporaryDirectory() as tmp:
        db=Commerce(Path(tmp)/'ui.db')
        with patch('aposira.commerce.Commerce',return_value=db):
            app=AppTest.from_file(Path(__file__).resolve().parents[1]/'aposira/app.py').run()
            app.session_state['cart_items']={'A':2}
            app.session_state['uploaded_video_first_frame_path']='old-branch-frame.png'
            app.text_input(key='branch_input').set_value('second-store').run()
            assert not app.exception
            assert 'cart_items' not in app.session_state
            assert 'uploaded_video_first_frame_path' not in app.session_state
