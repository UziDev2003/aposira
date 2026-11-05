"""Security event timeline shared with the embedded vision module."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

from aposira.vision.src.alerts.event_store import EventStore

EVENTS_DB = Path(__file__).resolve().parent / 'vision' / 'data' / 'events.sqlite3'


def render_timeline(org, branch):
    db_path = EVENTS_DB
    store = EventStore(db_path, org, branch)

    st.subheader('Security Event Timeline & Audit')
    st.caption('30-day event metadata retention. Scoped to active organization & branch. Alerts are observations for human review.')

    fcol1, fcol2, fcol3 = st.columns(3)
    start = fcol1.date_input('Event start (UTC)', datetime.now(timezone.utc).date() - timedelta(days=7), key='event_start')
    end = fcol2.date_input('Event end (UTC)', datetime.now(timezone.utc).date(), key='event_end')
    camera = fcol3.text_input('Camera ID filter', key='event_camera', placeholder='e.g., CAM-01')

    scol1, scol2 = st.columns([3, 1])
    search = scol1.text_input('Search message or zone', key='event_search', placeholder='Search by zone name or activity keywords...')
    unack = scol2.checkbox('Only unacknowledged', key='event_unack')

    if start > end:
        st.error('Event start date must precede end date.')
        return

    since_ts = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    until_ts = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() - 0.001

    rows = store.query(
        since=since_ts,
        until=until_ts,
        camera_id=camera.strip() if camera else None,
        search=search.strip() if search else None,
        include_acknowledged=not unack
    )

    if not rows:
        st.info('No matching events found for the selected filters. Process video or RTSP streams in Store Vision to generate security alerts.')
        return

    unack_count = sum(1 for r in rows if r.get('acknowledged_at') is None)
    unique_cams = len(set(r.get('camera_id') for r in rows if r.get('camera_id')))

    m1, m2, m3 = st.columns(3)
    m1.metric('Total Filtered Events', len(rows))
    m2.metric('Unacknowledged Alerts', unack_count)
    m3.metric('Cameras Active', unique_cams)

    display_rows = []
    for r in rows:
        ts = r.get('timestamp')
        time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if ts else ''
        sev = r.get('severity_score', 1)
        sev_label = "CRITICAL" if sev >= 4 else ("HIGH" if sev == 3 else ("MEDIUM" if sev == 2 else "LOW"))
        ack_at = r.get('acknowledged_at')
        ack_str = f"Yes ({datetime.fromtimestamp(ack_at, tz=timezone.utc).strftime('%m-%d %H:%M')})" if ack_at else "No"
        display_rows.append({
            'alert_id': r.get('alert_id'),
            'Time (UTC)': time_str,
            'Camera': r.get('camera_id', ''),
            'Alert Type': r.get('alert_type', '').replace('_', ' ').title(),
            'Severity': f"{sev_label} ({sev})",
            'Confidence': f"{float(r.get('confidence', 0)):.1%}",
            'Zone': r.get('zone_id') or 'General Area',
            'Message': r.get('message', ''),
            'Acknowledged': ack_str
        })

    frame = pd.DataFrame(rows)
    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df.drop(columns=['alert_id']), width='stretch', hide_index=True)

    col_dl, col_ack = st.columns([1, 1], gap='medium')
    with col_dl:
        st.download_button(
            'Export filtered events CSV',
            frame.to_csv(index=False),
            'aposira-events.csv',
            'text/csv'
        )

    with col_ack:
        with st.form('ack_event_form'):
            st.caption('Mark alert as acknowledged:')
            event_id = st.selectbox(
                'Select alert to acknowledge',
                [r['alert_id'] for r in rows],
                format_func=lambda aid: next(f"{r['alert_type']} - {r['camera_id']} ({aid[:8]})" for r in rows if r['alert_id'] == aid)
            )
            if st.form_submit_button('Acknowledge event'):
                store.acknowledge(event_id)
                st.success(f'Alert {event_id[:8]} acknowledged.')
                st.rerun()
