"""Defines the abstract base class for loggers."""

from abc import ABC, abstractmethod
from pathlib import Path

import altair as alt
import numpy as np

from hinky.logger.metrics import HostMetrics


class LoggerType(ABC):
  """Abstract base class for loggers."""

  log_dir: Path
  save_dir: Path
  version: str

  @abstractmethod
  def log_metric(
    self,
    metrics_dict: HostMetrics,
    step: int,
  ) -> None:
    """Logs a set of metrics at a given step.

    Args:
        metrics_dict: The dictionary of metrics to log.
        step: The current training step.
        log_postfix: An optional postfix to append to the metric keys.
    """
    pass

  @abstractmethod
  def finalize(self, status: str) -> None:
    """Finalizes the logging process.

    Args:
        status: The final status of the training (e.g., "success", "failure").
    """
    pass

  @abstractmethod
  def log_image(
    self,
    tag: str,
    image: np.ndarray,
    global_step: int,
    dataformats: str = "CHW",
  ) -> None:
    """Logs an image.

    Args:
        key: The key under which to log the image.
        image: The image to log, as a NumPy array.
        global_step: The current training step.
        dataformats: The data format of the image (e.g., "CHW", "HWC").
        log_postfix: An optional postfix to append to the tag.
    """
    pass

  @abstractmethod
  def log_figure(
    self,
    tag: str,
    figure: alt.Chart,
    global_step: int,
  ) -> None:
    """Logs a Bokeh figure.

    Args:
        key: The key under which to log the figure.
        figure: The Bokeh figure to log.
        global_step: The current training step.
        log_postfix: An optional postfix to append to the tag.
    """
    pass

  @abstractmethod
  def log_embedding(
    self,
    tag: str,
    mat: np.ndarray,
    metadata: list[str] | None,
    label_img: np.ndarray | None,
    global_step: int,
  ) -> None:
    """Logs an embedding.

    Args:
        key: The key under which to log the embedding.
        mat: The embedding matrix to log, as a NumPy array.
        metadata: A list of metadata strings corresponding to each embedding vector.
        label_img: An array of images corresponding to each embedding vector, for visualization.
        global_step: The current training step.
        log_postfix: An optional postfix to append to the tag.
    """
    pass

  @abstractmethod
  def log_hyperparams(
    self,
    params_dict: dict[str, str | int | float],
  ) -> None:
    """Logs hyperparameters.

    Args:
        params_dict: A dictionary of hyperparameters to log.
    """
