"""Exercise video callbacks without loading model weights or a Streamlit page."""
import ast
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aposira.vision.src.alerts.run_store import RunStore


def pipeline(frames=2, detector_error=None):
    source = Path('aposira/vision/streamlit_app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'process_video_batch')
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {1: frames, 2: 25, 3: 960, 4: 540}[prop]
    capture.read.side_effect = [(True, object())] * frames + [(False, None)]
    writer = Mock()
    writer.isOpened.return_value = True
    cv = SimpleNamespace(VideoCapture=Mock(return_value=capture),
                         VideoWriter=Mock(return_value=writer),
                         VideoWriter_fourcc=lambda *args: 0,
                         resize=lambda frame, size: frame,
                         CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2,
                         CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
    namespace = dict(cv2=cv, time=time, logging=logging, RETAIL_CLASSES=[0],
                     _associate_carried_objects=lambda *args: {},
                     _draw_rich_live_annotations=lambda **kwargs: kwargs['frame'],
                     _ensure_h264_compatibility=lambda path: None)
    code = 'from __future__ import annotations\n' + ast.get_source_segment(source, function)
    exec(compile(code, '<video pipeline>', 'exec'), namespace)
    detector = Mock()
    detector.detect.return_value = []
    detector.detect.side_effect = detector_error
    tracker = Mock()
    tracker.update.return_value = SimpleNamespace(tracks=[])
    analytics = Mock(zone_occ={})
    heatmap = Mock(total_footfall=0)
    behavior = Mock()
    behavior._check_congestion.return_value = []
    arguments = dict(video_path='input.mp4', output_path='output.mp4', det=detector,
                     trk=tracker, zones=[], zA=analytics, hm=heatmap, am=Mock(),
                     sus_detector=None, bhv=behavior, show_hm=False, enable_pose=False,
                     preproc=None, bg_sub=None, line_det=None, queue_det=None,
                     olb_det=None, orm_det=None, obj_shelves=None, face_blur=None,
                     video_clip=None, gest_d=None, fall_d=None, fight_d=None)
    return namespace['process_video_batch'], arguments, capture, writer


def test_preview_and_progress_arrive_before_output_finishes():
    process, arguments, capture, writer = pipeline()
    previews = []
    progress = []
    arguments['frame_callback'] = lambda frame, info: previews.append(
        (info['frame'], writer.release.called))
    arguments['progress_callback'] = lambda percent, info: progress.append(info['frame'])
    result = process(**arguments)
    assert previews[0] == (1, False)
    assert progress == [1, 2]
    assert result['processed_frames'] == 2
    capture.release.assert_called_once()
    writer.release.assert_called_once()


def test_model_failure_releases_video_resources():
    process, arguments, capture, writer = pipeline(detector_error=RuntimeError('GPU failed'))
    with pytest.raises(RuntimeError, match='GPU failed'):
        process(**arguments)
    capture.release.assert_called_once()
    writer.release.assert_called_once()


def test_optional_stage_errors_are_reported_once_and_processing_continues():
    process, arguments, _, _ = pipeline()
    arguments['preproc'] = Mock(side_effect=ValueError('invalid frame setting'))
    errors = []
    arguments['error_callback'] = lambda *args: errors.append(args)
    assert process(**arguments)['processed_frames'] == 2
    assert errors == [('Frame preprocessing', 1, 'invalid frame setting')]


def test_preview_failure_releases_video_resources():
    process, arguments, capture, writer = pipeline()
    arguments['frame_callback'] = Mock(side_effect=RuntimeError('Preview disconnected'))
    with pytest.raises(RuntimeError, match='Preview disconnected'):
        process(**arguments)
    capture.release.assert_called_once()
    writer.release.assert_called_once()


def test_real_video_decoding_sends_frames_before_completion(tmp_path):
    cv = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    source = tmp_path / 'input.mp4'
    output = tmp_path / 'output.mp4'
    writer = cv.VideoWriter(str(source), cv.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    assert writer.isOpened()
    try:
        for value in (0, 60, 120, 180):
            writer.write(np.full((120, 160, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    process, arguments, _, _ = pipeline()
    process.__globals__['cv2'] = cv
    arguments.update(video_path=str(source), output_path=str(output))
    previews = []
    arguments['frame_callback'] = lambda frame, info: previews.append(frame.shape)
    assert process(**arguments)['processed_frames'] == 4
    assert previews[0] == (540, 960, 3)
    assert output.stat().st_size > 0


def test_run_history_is_durable_scoped_and_records_failure(tmp_path):
    path = tmp_path / 'events.sqlite3'
    store = RunStore(path, 'shop', 'branch')
    run = store.start('CAM-2', 'sample.mp4')
    store.update(run, 7, 50)
    store.error(run, 'Detection', 7, 'GPU failed')
    store.update(run, 7, 50, status='failed', error='GPU failed')
    other = RunStore(path, 'shop', 'other')
    other.update(run, status='completed')
    other.error(run, 'Detection', 8, 'wrong branch')
    assert other.recent() == []
    assert other.errors(run) == []
    reopened = RunStore(path, 'shop', 'branch')
    assert reopened.recent()[0]['status'] == 'failed'
    assert reopened.recent()[0]['frame'] == 7
    assert len(reopened.errors(run)) == 1
    assert reopened.errors(run)[0]['message'] == 'GPU failed'
