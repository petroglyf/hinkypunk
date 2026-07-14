""""Callbacks for monitoring training, such as learning rate and gradient spikes."""
import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import optax
from pydantic import BaseModel, Field

from hinky.callbacks.callback import Callback, CallbackConfig

from hinky.trainer.trainer import TrainerModule

_logger = logging.getLogger(__name__)

class GradientSpikeConfig(BaseModel):
  """Configuration for GradientSpikeMonitor callback."""

  threshold: float = Field(
    default=2.0,
    gt=0.0,
    description="Threshold for detecting gradient spikes.",
  )

  log_to_disk: bool = Field(
    default=False,
    description="Whether to log the gradient norms and losses to disk as a .npz file.",
  )

  ema_decay: float = Field(
    default=0.995,
    gt=0.0,
    lt=1.0,
    description="Decay for the exponential moving average (EMA) used to detect spikes.",
  )


class LearningRateMonitor(Callback):
  """Callback to monitor the learning rate."""

  def __init__(
    self,
    callback_config: CallbackConfig,
    trainer: TrainerModule,
  ) -> None:
    """Initializes the LearningRateMonitor callback."""
    super().__init__(callback_config)
    self.log_dir = trainer.log_dir
    self.trainer = trainer

  def on_filtered_training_epoch_start(self, epoch_idx: int) -> None:
    # Log the learning rate at the beginning of the first epoch.
    if epoch_idx == 1:
      self._log_lr(epoch_idx - 1)

  def on_filtered_training_epoch_end(self, train_metrics: dict[str, float], epoch_idx: int) -> None:  # noqa: ARG002
    self._log_lr(epoch_idx)

  def _log_lr(self, epoch_idx: int) -> None:
    """Logs the learning rate.

    Args:
        epoch_idx: Index of the current epoch. Used as logging step.
    """
    schedule = self.trainer.lr_schedule
    if schedule is None:
      _logger.warning("No learning rate schedule found.")
      return
    opt_state = [
      s for s in self.trainer.state.opt_state[-1] if isinstance(s, optax.ScaleByScheduleState)
    ]
    if len(opt_state) == 0:
      _logger.warning("No state of a learning rate schedule found.")
      return
    if len(opt_state) > 1:
      _logger.warning("Found multiple states of a learning rate schedule. Using the last one.")
    step = opt_state[-1].count
    lr = schedule(step)
    self.trainer.logger.log_scalar("optimizer/lr", lr, epoch_idx)


class GradientSpikeMonitor(Callback):
  """Callback to monitor gradient spikes."""

  def __init__(
    self, config: CallbackConfig, gs_config: GradientSpikeConfig, trainer: TrainerModule,
  ) -> None:
    """Initializes the GradientSpikeMonitor callback."""
    super().__init__(config)
    if not trainer.trainer_config.log_grad_norm:
      msg = "log_grad_norm must be True to use GradientSpikeMonitor."
      raise ValueError(msg)
    self.log_dir = trainer.log_dir
    self.threshold = gs_config.threshold
    self.log_to_disk = gs_config.log_to_disk
    self.ema_decay = gs_config.ema_decay
    self.max_elements = int(np.log(1e-3) / np.log(self.ema_decay))
    self.trainer = trainer

  def on_training_start(self) -> None:
    self.grad_norms_buffer = []
    self.losses_buffer = []
    self.grad_norms = np.array([], dtype=np.float64)
    self.losses = np.array([], dtype=np.float64)

  def on_training_step(
    self,
    train_metrics: Any,
    epoch_idx: int,  # noqa: ARG002
    global_step: int,  # noqa: ARG002
  ) -> None:
    step_metrics = train_metrics
    if "optimizer/grad_global_norm" not in step_metrics or "loss" not in step_metrics:
      _logger.warning(
        _:= f"Step metrics must contain 'optimizer/grad_global_norm' and 'loss' keys, but got {list(step_metrics.keys())}.",  # noqa: E501
      )
      return
    self.grad_norms_buffer.append(self._metric_to_val(step_metrics["optimizer/grad_global_norm"]))
    self.losses_buffer.append(self._metric_to_val(step_metrics["loss"]))

  def _metric_to_val(self, metric: dict[str, Any]) -> dict[str, Any] | float:
    if isinstance(metric, dict):
      if "value" not in metric:
         _logger.warning(
           _:= f"Metric dict must contain a 'value' key, but got {list(metric.keys())}.",
         )
      return metric["value"]

    # else
    return metric

  def on_filtered_training_epoch_end(self, train_metrics: dict[str, Any], epoch_idx: int) -> None:
    del train_metrics
    epoch_grad_norms = np.asarray(jax.device_get(self.grad_norms_buffer))
    epoch_losses = np.asarray(jax.device_get(self.losses_buffer))
    self.grad_norms = np.concatenate([self.grad_norms, epoch_grad_norms])
    self.losses = np.concatenate([self.losses, epoch_losses])
    self.grad_norms_buffer = []
    self.losses_buffer = []
    if self.log_to_disk:
      np.savez(
        Path(self.log_dir) / "gradient_spikes.npz",
        grad_norms=self.grad_norms,
        losses=self.losses,
      )
    self.log_gradients_spikes(num_elements=epoch_losses.shape[0], epoch_idx=epoch_idx)

  def log_gradients_spikes(self, num_elements: int, epoch_idx: int) -> None:
    grad_norms = self.grad_norms
    losses = self.losses
    if num_elements + self.max_elements < self.grad_norms.shape[0]:
      grad_norms = grad_norms[-(num_elements + self.max_elements) :]
      losses = losses[-(num_elements + self.max_elements) :]
    # Calculate EMA by giving each element the weight for the final EMA element.
    weights = np.zeros_like(grad_norms) + self.ema_decay
    weights = np.power(weights, np.flip(np.arange(grad_norms.shape[0])))
    # Normalize all means with respect to all previous weights.
    weight_cumsum = np.cumsum(weights)
    grad_norms_cumsum = np.cumsum(grad_norms * weights)
    losses_cumsum = np.cumsum(losses * weights)
    grad_norms_ema = grad_norms_cumsum / weight_cumsum
    losses_ema = losses_cumsum / weight_cumsum
    # Check for elements that are spikes compared to the previous EMA.
    grad_norms_spike = (
      grad_norms[-min(num_elements, grad_norms.shape[0] - 1) :]
      > self.threshold * grad_norms_ema[-num_elements - 1 : -1]
    )
    losses_spike = (
      losses[-min(num_elements, losses.shape[0] - 1) :]
      > self.threshold * losses_ema[-num_elements - 1 : -1]
    )
    # Remove spikes that directly come after spikes.
    grad_norms_spike = grad_norms_spike & np.logical_not(
      np.concatenate([[False], grad_norms_spike[:-1]]),
    )
    losses_spike = losses_spike & np.logical_not(np.concatenate([[False], losses_spike[:-1]]))
    # Log the spikes.
    grad_norms_num_spikes = np.sum(grad_norms_spike)
    losses_num_spikes = np.sum(losses_spike)
    self.trainer.logger.log_scalar("optimizer/spikes_grad_norms", grad_norms_num_spikes, epoch_idx)
    self.trainer.logger.log_scalar("optimizer/spikes_losses", losses_num_spikes, epoch_idx)
    synchronous_spikes = np.sum(grad_norms_spike & losses_spike)
    self.trainer.logger.log_scalar("optimizer/spikes_synchronous", synchronous_spikes, epoch_idx)
    grad_spike_before_loss = np.sum(grad_norms_spike & np.concatenate([[False], losses_spike[:-1]]))
    self.trainer.logger.log_scalar(
      "optimizer/spikes_grad_before_loss", grad_spike_before_loss, epoch_idx,
    )
    loss_spike_before_grad = np.sum(losses_spike & np.concatenate([[False], grad_norms_spike[:-1]]))
    self.trainer.logger.log_scalar(
      "optimizer/spikes_loss_before_grad", loss_spike_before_grad, epoch_idx,
    )
