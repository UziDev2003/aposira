"""APOSIRA Store Vision; heavy inference dependencies load on demand."""
from importlib import import_module
__version__ = '1.0.0'
_EXPORTS = {
    'YOLODetector': 'aposira.vision.src.detector.yolo_detector',
    'ByteTracker': 'aposira.vision.src.tracker.bytetrack',
    'ZoneAnalytics': 'aposira.vision.src.analytics.zone_analytics',
    'HeatmapGenerator': 'aposira.vision.src.analytics.heatmap',
    'BehaviorDetector': 'aposira.vision.src.analytics.behavior_detector',
    'AlertManager': 'aposira.vision.src.alerts.alert_manager',
    'StreamManager': 'aposira.vision.src.stream.stream_manager',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]),name)
    raise AttributeError(name)
