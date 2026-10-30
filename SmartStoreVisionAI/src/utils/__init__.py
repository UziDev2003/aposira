"""Utilities with lazy imports for lightweight configuration access."""
from importlib import import_module
_EXPORTS = {'load_config':'src.utils.config','save_config':'src.utils.config','draw_text_with_background':'src.utils.helpers'}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]),name)
    raise AttributeError(name)
