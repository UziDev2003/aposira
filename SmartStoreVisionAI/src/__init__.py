"""APOSIRA Store Vision; heavy inference dependencies load on demand."""
from importlib import import_module
__version__ = '1.0.0'
_EXPORTS = {
    'YOLODetector': 'src.detector.yolo_detector',
    'ByteTracker': 'src.tracker.bytetrack',
    'ZoneAnalytics': 'src.analytics.zone_analytics',
    'HeatmapGenerator': 'src.analytics.heatmap',
    'BehaviorDetector': 'src.analytics.behavior_detector',
    'AlertManager': 'src.alerts.alert_manager',
    'StreamManager': 'src.stream.stream_manager',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]),name)
    raise AttributeError(name)
