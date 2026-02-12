import time

import jax
import optax
from absl import logging
from pydantic import BaseModel, Field

from typing import Any, TYPE_CHECKING
from jax_trainer.callbacks.callback import Callback, CallbackConfig
from jax_trainer.trainer.trainer import TrainerModule
from jax_trainer.trainer.state import TrainState


class JAXConfig(BaseModel):
  every_n_minutes: int = Field(
    60, gt=0, description="Interval in minutes between profiling sessions."
  )
  first_step: int = Field(
    10, gt=0, description="Step to start the first profiling session."
  )
  profile_n_steps: int = Field(
    5, gt=0, description="Number of steps to profile during each session."
  )


class JAXProfiler(Callback):
  """Callback to profile model training steps."""

  def __init__(
    self,
    params_config: JAXConfig,
    callback_config: CallbackConfig,
    trainer: TrainerModule,
  ):
    super().__init__(callback_config)
    self.log_dir = trainer.trainer_config.logger.log_dir
    self.every_n_minutes = params_config.every_n_minutes
    self.first_step = params_config.first_step
    self.profiler_n_steps = params_config.profile_n_steps
    self.profiler_active = False
    self.profiler_last_time = float("inf")

  def on_training_start(self):
    self.profiler_active = False
    self.profiler_last_time = time.time()

  def on_training_step(
    self, train_metrics, train_state: TrainState, epoch_idx: int, global_step: int
  ):
    # def on_training_step(self, trainer_state, step_metrics, epoch_idx, step_idx):
    if self.profiler_active:
      if global_step >= self.profile_start_step + self.profiler_n_steps:
        self.stop_trace(train_state)
    elif (global_step == self.first_step) or (
      time.time() - self.profiler_last_time > self.every_n_minutes * 60
    ):
      self.start_trace(global_step)

  def on_training_epoch_end(
    self,
    train_state: TrainState,
    train_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    self.stop_trace(train_state)

  def start_trace(self, step_idx):
    if not self.profiler_active:
      logging.info(f"Starting trace at step {step_idx}.")
      jax.profiler.start_trace(self.log_dir)
      self.profiler_active = True
      self.profile_start_step = step_idx
    else:
      logging.warning("Trace already active.")

  def stop_trace(self, train_state: TrainState):
    if self.profiler_active:
      logging.info("Stopping trace")
      jax.tree_map(lambda x: x.block_until_ready(), train_state.params)
      jax.profiler.stop_trace()
      self.profiler_last_time = time.time()
      self.profiler_active = False

  @classmethod
  def encapsulate_config(cls, options: dict[str, Any]) -> JAXConfig:
    """Translates a dictionary of options into a JAXConfig pydantic model."""
    return JAXConfig(**options)
