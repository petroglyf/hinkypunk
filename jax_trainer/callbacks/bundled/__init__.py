"""Bundled callbacks for common use cases, such as classification and regression."""
from jax_trainer.callbacks.bundled.classification import ConfusionMatrixCallback  # noqa: F401

# from jax_trainer.callbacks.bundled.monitor import (
#   # GradientSpikeMonitor,
#   LearningRateMonitor,
# )
from jax_trainer.callbacks.bundled.profiler import JAXProfiler  # noqa: F401

__all__ = []
