"""Callback to profile model training steps using JAX's built-in profiler."""
import time
from typing import Any

import jax
from absl import logging
from pydantic import BaseModel, Field

from hinky.callbacks.callback import Callback, CallbackConfig
from hinky.trainer.trainer import TrainerModule


class JAXConfig(BaseModel):
  every_n_minutes: int = Field(
    60, gt=0, description="Interval in minutes between profiling sessions.",
  )
  first_step: int = Field(10, gt=0, description="Step to start the first profiling session.")
  profile_n_steps: int = Field(
    5, gt=0, description="Number of steps to profile during each session.",
  )


class JAXProfiler(Callback):
  """Callback to profile model training steps."""

  def __init__(
    self,
    params_config: JAXConfig,
    callback_config: CallbackConfig,
    trainer: TrainerModule,
  ) -> None:
    """Initializes the JAXProfiler callback."""
    super().__init__(callback_config)
    self.log_dir = trainer.trainer_config.logger.log_dir
    self.every_n_minutes = params_config.every_n_minutes
    self.first_step = params_config.first_step
    self.profiler_n_steps = params_config.profile_n_steps
    self.profiler_active = False
    self.profiler_last_time = float("inf")
    self.trainer = trainer

  def on_training_start(self) -> None:
    self.profiler_active = False
    self.profiler_last_time = time.time()

  def on_training_step(
    self,
    train_metrics: Any,  # noqa: ARG002
    epoch_idx: int,  # noqa: ARG002
    global_step: int,
  ) -> None:
    if self.profiler_active:
      if global_step >= self.profile_start_step + self.profiler_n_steps:
        self.stop_trace()
    elif (global_step == self.first_step) or (
      time.time() - self.profiler_last_time > self.every_n_minutes * 60
    ):
      self.start_trace(global_step)

  def on_training_epoch_end(
    self,
    train_metrics: dict[str, float],  # noqa: ARG002
    epoch_idx: int,  # noqa: ARG002
  ) -> None:
    self.stop_trace()

  def start_trace(self, step_idx: int) -> None:
    if not self.profiler_active:
      logging.info(f"Starting trace at step {step_idx}.")
      jax.profiler.start_trace(self.log_dir)
      self.profiler_active = True
      self.profile_start_step = step_idx
    else:
      logging.warning("Trace already active.")

  def stop_trace(self) -> None:
    if self.profiler_active:
      logging.info("Stopping trace")
      state = self.trainer.state
      jax.tree.map(lambda x: x.block_until_ready(), state)
      jax.profiler.stop_trace()
      self.profiler_last_time = time.time()
      self.profiler_active = False

  @classmethod
  def encapsulate_config(cls, options: dict[str, Any]) -> JAXConfig:
    """Translates a dictionary of options into a JAXConfig pydantic model."""
    return JAXConfig(**options)
