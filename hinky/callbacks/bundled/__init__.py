"""Bundled callbacks for common use cases, such as classification and regression."""
from hinky.callbacks.bundled.classification import ConfusionMatrixCallback  # noqa: F401

# from hinky.callbacks.bundled.monitor import (
#   # GradientSpikeMonitor,
#   LearningRateMonitor,
# )
from hinky.callbacks.bundled.profiler import JAXProfiler  # noqa: F401

__all__ = []
