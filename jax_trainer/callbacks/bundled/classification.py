import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
from pydantic import BaseModel, Field

from jax_trainer.callbacks.callback import Callback, CallbackConfig
from jax_trainer.trainer.trainer import TrainerModule


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
  """Callback to visualize the confusion matrix."""

  def __init__(
    self,
    params_config: ConfusionMatrixConfig,
    callback_config: CallbackConfig,
    trainer: TrainerModule,
  ):
    super().__init__(callback_config)
    self.log_dir = trainer.trainer_config.logger.log_dir
    self.class_names = trainer.dataset_config.class_names
    self.cf_config = params_config

  def on_filtered_validation_epoch_end(
    self,
    eval_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    return self._visualize_confusion_matrix(eval_metrics, epoch_idx)

  def _on_test_epoch_end(self, test_metrics: dict[str, float], epoch_idx: int) -> None:
    """Logs confusion matrix at the end of testing epoch."""
    return self._visualize_confusion_matrix(test_metrics, epoch_idx)

  def _visualize_confusion_matrix(self, metrics: dict[str, Any], epoch_idx: int) -> None:
    """Visualizes and logs the confusion matrix."""
    conf_key = [k for k in metrics if k.endswith("conf_matrix")]
    if len(conf_key) == 0:
      logging.warning(f"Confusion matrix not found in eval metrics, only found {metrics.keys()}.")
      return
    conf_key = conf_key[0]
    conf_matrix = metrics[conf_key]
    if self.cf_config.normalize:
      conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
      format = self.cf_config.format
    else:
      format = self.cf_config.format
    fig, ax = plt.subplots(figsize=self.cf_config.figsize, dpi=self.cf_config.dpi)
    # sns.heatmap(conf_matrix, annot=True, cmap=self.cf_config.cmap, ax=ax, fmt=format)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion matrix")
    ax.set_xticks(np.arange(conf_matrix.shape[0]) + 0.5)
    ax.set_yticks(np.arange(conf_matrix.shape[1]) + 0.5)

    ax.set_xticklabels(self.class_names)
    ax.set_yticklabels(self.class_names)

    fig.tight_layout()
    self.trainer.logger.log_figure("confusion_matrix", fig, epoch_idx)

  @classmethod
  def encapsulate_config(cls, options: dict[str, Any]) -> ConfusionMatrixConfig:
    """Translates a dictionary of options into a ConfusionMatrixConfig pydantic model."""
    return ConfusionMatrixConfig(**options)
