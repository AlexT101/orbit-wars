"""JAX engine candidate adapter for the parity harness."""

from __future__ import annotations

from bots.mine.jax_engine import JaxEngine as _JaxEngine

JaxEngine = _JaxEngine

__all__ = ["JaxEngine"]
