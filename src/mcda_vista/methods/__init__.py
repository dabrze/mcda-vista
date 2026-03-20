"""MCDA method adapters for VISTA.

Use :func:`get_method`, :func:`list_methods`, and :func:`register_method`
to interact with the built-in and user-defined method registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcda_vista.methods.base import MethodAdapter, handle_pydecision_warnings

if TYPE_CHECKING:
    pass

__all__ = [
    "MethodAdapter",
    "get_method",
    "handle_pydecision_warnings",
    "list_methods",
    "register_method",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MethodAdapter] = {}


def register_method(name: str):
    """Class decorator that registers a method adapter under *name*.

    Example::

        @register_method("saw")
        class SAWAdapter:
            name = "saw"
            display_name = "SAW"
            ...
    """

    def decorator(cls):
        _REGISTRY[name] = cls()
        return cls

    return decorator


def get_method(name: str) -> MethodAdapter:
    """Look up a registered method adapter by *name*.

    Raises :class:`KeyError` if *name* is not found.
    """
    _ensure_builtins_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown method '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """Return names of all registered methods."""
    _ensure_builtins_loaded()
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Lazy loading of built-in adapters
# ---------------------------------------------------------------------------

_builtins_loaded = False


def _ensure_builtins_loaded() -> None:
    """Import all built-in adapter modules so they register themselves."""
    global _builtins_loaded
    if not _builtins_loaded:
        # Import triggers @register_method decorators in each module.
        from mcda_vista.methods import (  # noqa: F401
            assess,
            borda,
            electre,
            etopsis,
            macbeth,
            oreste,
            promethee,
            regime,
            saw,
            spotis,
            topsis,
            uta,
            vikor,
        )

        _builtins_loaded = True