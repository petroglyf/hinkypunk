"""Module for callback implementations in the JAX Trainer framework."""

from jax_trainer.callbacks.callback import Callback
from jax_trainer.callbacks.config import CallbackConfig

__all__ = [
  "Callback",
  "CallbackConfig",
]
