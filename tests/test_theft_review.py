"""Evidence sequences and false-positive checks for theft-related review alerts."""
import time

import numpy as np
import pytest

from aposira.vision.src.alerts.alert_manager import AlertManager
from aposira.vision.src.analytics.suspicious_activity import (
    ActivityType, CooldownTracker, PossibleConcealmentDetector,
    PossibleUnpaidExitDetector, ShelfTamperingDetector,
    SuspiciousActivityDetector,
)
from aposira.vision.src.analytics.zone_analytics import Zone
from aposira.vision.src.detector.yolo_detector import Detection
from aposira.vision.src.tracker.bytetrack import Track
from aposira.vision.src.utils.config_schema import (
    PossibleConcealmentConfig, PossibleUnpaidExitConfig, SuspiciousActivityConfig,
)


def scene():
    return {
        'shelf': Zone('shelf', 'Shelf', 'shelf', [(0, 0), (90, 0), (90, 100), (0, 100)]),
        'checkout': Zone('checkout', 'Till', 'checkout', [(125, 0), (175, 0), (175, 100), (125, 100)]),
        'exit': Zone('exit', 'Door', 'exit', [(200, 0), (300, 0), (300, 100), (200, 100)]),
    }


def observations(start, checkout=False, preexisting=False, carry_at_exit=True):
    positions = [30, 30, 100, 135 if checkout else 180, 210, 230]
    for index, x in enumerate(positions):
        ts = start + (0 if index == 0 else 2 + index * 0.5)
        person = Track(1, 0, 0.95, (x, 20, x + 30, 95), ts)
        has_object = (index > 0 or preexisting) and (index < 5 or carry_at_exit)
        objects = [Detection(39, 'bottle', 0.95, (x + 8, 40, x + 20, 60))] if has_object else []
        yield person, objects, ts


@pytest.mark.parametrize('checkout,preexisting,carry_at_exit,expected', [
    (False, False, True, 1),
    (True, False, True, 0),
    (False, True, True, 0),
    (False, False, False, 0),
])
def test_exit_sequence_requires_new_persistent_object_and_no_checkout(
        checkout, preexisting, carry_at_exit, expected):
    detector = PossibleUnpaidExitDetector(PossibleUnpaidExitConfig())
    cooldowns = CooldownTracker()
    events = []
    for person, objects, ts in observations(1000, checkout, preexisting, carry_at_exit):
        events.extend(detector.update([person], scene(), {'shelf': {1: 1000}}, objects,
                                      ts, cooldowns, 'CAM-2'))
    assert len(events) == expected
    if events:
        assert events[0].activity_type == ActivityType.POSSIBLE_UNPAID_EXIT
        assert events[0].evidence['payment_verified'] is False
        assert events[0].evidence['carry_frames'] >= 5


def test_missing_checkout_zone_cannot_produce_unpaid_exit():
    zones = scene()
    zones.pop('checkout')
    detector = PossibleUnpaidExitDetector(PossibleUnpaidExitConfig())
    for person, objects, ts in observations(1000):
        assert detector.update([person], zones, {'shelf': {1: 1000}}, objects,
                               ts, CooldownTracker(), 'CAM-2') == []


def test_small_local_shelf_motion_triggers_and_static_frames_do_not():
    cfg = SuspiciousActivityConfig().detectors.shelf_tampering
    cfg.min_interaction_seconds = 1
    cfg.confirmation_frames = 2
    zone = Zone('shelf', 'Large shelf', 'shelf', [(0, 0), (400, 0), (400, 400), (0, 400)])
    person = Track(1, 0, 0.95, (20, 20, 60, 60), 1000)
    detector = ShelfTamperingDetector(cfg)
    events = []
    for index in range(4):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        if index % 2:
            frame[20:60, 20:60] = 255
        events.extend(detector.update(frame, [person], {'shelf': zone}, {'shelf': {1: 995}},
                                      {}, 1000 + index, CooldownTracker(), 'CAM-2'))
    assert events
    assert events[0].evidence['motion_intensity'] > cfg.min_motion_intensity
    detector = ShelfTamperingDetector(cfg)
    for index in range(4):
        assert detector.update(np.zeros((400, 400, 3), dtype=np.uint8), [person],
                               {'shelf': zone}, {'shelf': {1: 995}}, {}, 1000 + index,
                               CooldownTracker(), 'CAM-2') == []


def test_review_event_passes_filters_and_persists_in_camera_timeline(tmp_path):
    cfg = SuspiciousActivityConfig()
    for name in ('loitering', 'restricted_zone', 'shelf_tampering', 'abnormal_shelf',
                 'unattended_checkout', 'movement', 'pose'):
        getattr(cfg.detectors, name).enabled = False
    detector = SuspiciousActivityDetector(cfg)
    manager = AlertManager(snapshot_dir=tmp_path / 'snapshots', event_db=tmp_path / 'events.db',
                           organization_id='shop', store_id='branch')
    start = time.time() - 10
    events = []
    for person, objects, ts in observations(start):
        events.extend(detector.detect([person], scene(), {'shelf': {1: start}},
                                      timestamp=ts, detections=objects, camera_id='CAM-2'))
    assert len(events) == 1
    manager.create_from_activity(events[0], 'CAM-2')
    saved = manager.event_store.query(camera_id='CAM-2')
    assert saved[0]['alert_type'] == 'possible_unpaid_exit'
    assert saved[0]['evidence']['carried_objects'] == ['bottle']


def concealment_sequence(positions, drop_track=False):
    detector = PossibleConcealmentDetector(PossibleConcealmentConfig())
    shelf = Zone('shelf', 'Shelf', 'shelf', [(110, 50), (160, 50), (160, 200), (110, 200)])
    events = []
    cooldowns = CooldownTracker()
    for index, x in enumerate(positions):
        ts = 1000 + index * 0.3
        person = Track(1, 0, 0.95, (20, 20, 120, 220), ts)
        objects = [] if x is None else [Detection(39, 'bottle', 0.95, (x - 5, 115, x + 5, 125))]
        tracks = [] if drop_track and index == 3 else [person]
        events.extend(detector.update(tracks, {'shelf': shelf}, objects, ts, cooldowns, 'CAM-2'))
    return events


def test_shelf_item_moves_to_torso_then_stays_obscured():
    events = concealment_sequence([115, 70, None, None, None, None, None])
    assert len(events) == 1
    assert events[0].activity_type == ActivityType.POSSIBLE_CONCEALMENT
    assert events[0].evidence['person_still_tracked'] is True
    assert events[0].evidence['requires_review'] is True
    assert events[0].evidence['hidden_frames'] == 5


@pytest.mark.parametrize('positions', [
    [115, 70, None, None, 70],  # Brief occlusion followed by visible item.
    [70, 70, None, None, None, None, None],  # No observed shelf-to-body movement.
    [115, 115, None, None, None, None, None],  # Item disappeared at the shelf.
    [115, 70, None],  # Insufficient evidence duration.
])
def test_concealment_suppresses_incomplete_or_reappearing_evidence(positions):
    assert concealment_sequence(positions) == []


def test_losing_the_person_track_resets_concealment_sequence():
    assert concealment_sequence([115, 70, None, None, None, None, None], drop_track=True) == []


def test_item_moving_into_a_visible_bag_can_trigger_review():
    detector = PossibleConcealmentDetector(PossibleConcealmentConfig())
    shelf = Zone('shelf', 'Shelf', 'shelf', [(160, 50), (190, 50), (190, 200), (160, 200)])
    bag = Detection(26, 'handbag', 0.95, (125, 105, 155, 170))
    cooldowns = CooldownTracker()
    events = []
    for index, x in enumerate([165, 140, None, None, None, None, None]):
        ts = 1000 + index * 0.3
        person = Track(1, 0, 0.95, (20, 20, 120, 220), ts)
        objects = [bag]
        if x is not None:
            objects.append(Detection(39, 'bottle', 0.95, (x - 5, 115, x + 5, 125)))
        events.extend(detector.update([person], {'shelf': shelf}, objects, ts, cooldowns, 'CAM-2'))
    assert len(events) == 1
    assert events[0].activity_type == ActivityType.POSSIBLE_CONCEALMENT
