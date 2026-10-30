"""Alert module; event storage remains usable without computer vision packages."""
__all__ = ['AlertManager']


def __getattr__(name):
    if name == 'AlertManager':
        from .alert_manager import AlertManager
        return AlertManager
    raise AttributeError(name)
