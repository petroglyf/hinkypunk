"""Callbacks for classification tasks, including confusion matrix visualization."""
import logging
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from hinky.callbacks.callback import Callback, CallbackConfig
from hinky.trainer.trainer import TrainerModule

_logger = logging.getLogger(__name__)


class ConfusionMatrixConfig(BaseModel):
  """Configuration for ConfusionMatrixCallback."""

  normalize: bool = Field(
    default=False,
    description="Whether to normalize the confusion matrix.",
  )

  format: str = Field(
    default="d",
    description="Format for annotations in the confusion matrix.",
  )

  figsize: tuple[int, int] = Field(
    default=(8, 8),
    description="Figure size for the confusion matrix plot.",
  )

  dpi: int = Field(default=100, description="DPI for the confusion matrix plot.")

  cmap: str = Field(
    default="Blues",
    description="Colormap for the confusion matrix plot.",
  )


class ConfusionMatrixCallback(Callback):
  """Callback to visualize the confusion matrix.

  Expects the eval/test metrics dict passed to it to contain a key ending in
  "conf_matrix" whose value is an (n_classes, n_classes) array, e.g. as
  produced by a confusion-matrix metric registered on the trainer. The
  resulting heatmap is logged via the trainer's logger under the key
  "confusion_matrix".
  """

  def __init__(
    self,
    params_config: ConfusionMatrixConfig,
    callback_config: CallbackConfig,
    trainer: TrainerModule,
  ) -> None:
    """Initializes the callback from its config and the owning trainer.

    Args:
      params_config: Plot-specific settings (normalization, format, size, colormap).
      callback_config: Base callback settings (e.g. when it should fire).
      trainer: Trainer instance providing the logger, dataset class names, and log dir.
    """
    super().__init__(callback_config)
    self.log_dir = trainer.trainer_config.logger.log_dir
    self.class_names = trainer.dataset_config.class_names
    self.cf_config = params_config
    self.trainer = trainer

  def on_filtered_validation_epoch_end(
    self,
    eval_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Logs the confusion matrix at the end of a (filtered) validation epoch."""
    return self._visualize_confusion_matrix(eval_metrics, epoch_idx)

  def _on_test_epoch_end(self, test_metrics: dict[str, float], epoch_idx: int) -> None:
    """Logs confusion matrix at the end of testing epoch."""
    return self._visualize_confusion_matrix(test_metrics, epoch_idx)

  def _visualize_confusion_matrix(self, metrics: dict[str, Any], epoch_idx: int) -> None:
    """Builds a confusion-matrix heatmap from metrics and logs it via the trainer.

    Searches `metrics` for a key ending in "conf_matrix", optionally
    row-normalizes it, then renders an Altair heatmap (with per-cell value
    annotations) and hands it to `self.trainer.logger.log_figure`. If no such
    key is found, logs a warning and returns without raising.

    Args:
      metrics: Metrics dict for the epoch, expected to contain a
        "*conf_matrix" entry with an (n_classes, n_classes) array-like value.
      epoch_idx: Current epoch index, used to tag the logged figure.
    """
    conf_key = [k for k in metrics if k.endswith("conf_matrix")]
    if len(conf_key) == 0:
      _logger.warning(_ := f"Confusion matrix not found in eval metrics, only found {metrics.keys()}.")
      return
    conf_key = conf_key[0]
    conf_matrix = metrics[conf_key]
    if self.cf_config.normalize:
      conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    n = conf_matrix.shape[0]
    labels = self.class_names or [str(i) for i in range(n)]
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    df = pd.DataFrame({
      "True": [labels[i] for i in rows.flatten()],
      "Predicted": [labels[j] for j in cols.flatten()],
      "value": conf_matrix.flatten().tolist(),
    })
    width = self.cf_config.figsize[0] * self.cf_config.dpi
    height = self.cf_config.figsize[1] * self.cf_config.dpi
    heatmap = alt.Chart(df).mark_rect().encode(
      x=alt.X("Predicted:O", axis=alt.Axis(labelAngle=-45), title="Predicted labels"),
      y=alt.Y("True:O", title="True labels"),
      # pyrefly: ignore [bad-argument-type]
      color=alt.Color("value:Q", scale=alt.Scale(scheme=self.cf_config.cmap.lower())),
    )
    text = alt.Chart(df).mark_text().encode(
      x="Predicted:O",
      y="True:O",
      text=alt.Text("value:Q", format=self.cf_config.format),
    )
    chart = (heatmap + text).properties(width=width, height=height, title="Confusion matrix")
    self.trainer.logger.log_figure("confusion_matrix", chart, epoch_idx)

  @classmethod
  def encapsulate_config(cls, options: dict[str, Any]) -> ConfusionMatrixConfig:
    """Translates a dictionary of options into a ConfusionMatrixConfig pydantic model."""
    return ConfusionMatrixConfig(**options)
