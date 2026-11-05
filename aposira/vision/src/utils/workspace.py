"""Request-local configuration scope for embedded APOSIRA sessions."""
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
SCOPE=ContextVar('aposira_vision_scope',default=None)


def scope_directory():
    scope=SCOPE.get()
    return sha256(('\0'.join(scope)).encode()).hexdigest()[:24] if scope else None


def runtime_directory(base):
    scope=scope_directory()
    return Path(base)/scope if scope else Path(base)
