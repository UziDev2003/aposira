"""Timeline UI shared with the embedded vision module."""
from datetime import datetime, timedelta, timezone
import sys
import pandas as pd
import streamlit as st


def render_timeline(root, org, branch):
    sys.path.insert(0,str(root/'SmartStoreVisionAI'))
    from src.alerts.event_store import EventStore
    store=EventStore(root/'SmartStoreVisionAI/data/events.sqlite3',org,branch)
    st.subheader('Security Event Timeline')
    st.caption('30-day event metadata retention. Alerts are observations for human review, not proof of wrongdoing. Snapshot retention is configured separately.')
    start=st.date_input('Event start (UTC)',datetime.now(timezone.utc).date()-timedelta(days=7),key='event_start')
    end=st.date_input('Event end (UTC)',datetime.now(timezone.utc).date(),key='event_end')
    camera=st.text_input('Camera ID filter',key='event_camera')
    search=st.text_input('Search message / zone',key='event_search')
    unack=st.checkbox('Only unacknowledged',key='event_unack')
    if start>end:st.error('Start must precede end.');return
    rows=store.query(since=datetime.combine(start,datetime.min.time(),tzinfo=timezone.utc).timestamp(),
                     until=datetime.combine(end+timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc).timestamp()-.001,
                     camera_id=camera or None,search=search,include_acknowledged=not unack)
    if not rows:st.info('No matching events. Run a camera or video through Store Vision to record events.');return
    frame=pd.DataFrame(rows)
    st.dataframe(frame,width='stretch')
    st.download_button('Export filtered events CSV',frame.to_csv(index=False),'aposira-events.csv','text/csv')
    event_id=st.selectbox('Event to acknowledge',[r['alert_id'] for r in rows])
    if st.button('Acknowledge event'):
        store.acknowledge(event_id);st.rerun()
