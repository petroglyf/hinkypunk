# pyright: ignore[reportInvalidTypeForm]

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, Final, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import FrozenDict
from progress_table.progress_table import ProgressTable, TableProgressBar

from jax_trainer.callbacks.callback import Callback
from jax_trainer.logger import (
  HostMetrics,
  ImmutableMetrics,
)
from jax_trainer.logger.loggers import Logger

from .train_stepper import TrainStep

_logger = logging.getLogger(__name__)


class TrackerFn(Protocol):
  def __call__(
    self, progress_table: ProgressTable, iterator: Iterator, **kwargs: Any
  ) -> Iterator | TableProgressBar: ...


BatchToInputFn = Callable[[dict[str, jax.Array]], dict[str, Any]]


class EpochStep:
  """Class to encapsulate the logic for training a model for one epoch."""

  def __init__(
    self,
    tracker: TrackerFn,
    logger: Logger,
    train_step_callbacks: list[Callback[Any]],
    batch_to_input_fn: BatchToInputFn,
    optimizer_and_model: nnx.ModelAndOptimizer,
    train_step: TrainStep,
    *,
    enable_progress_bar: bool,
  ) -> None:
    self.total_epochs_taken = 0
    self.tracker: Final[TrackerFn] = tracker
    self.logger: Final[Logger] = logger
    self.enable_progress_bar: Final[bool] = enable_progress_bar
    self.optimizer_and_model = optimizer_and_model
    self.train_step: Final[TrainStep] = train_step
    self.train_metric_shapes = None
    self.train_step_callbacks: Final[list[Callback[Any]]] = train_step_callbacks
    self.batch_to_input_fn: Final[BatchToInputFn] = batch_to_input_fn

  def init_train_metrics(
    self,
    batch: dict[str, jax.Array] | None = None,
    *,
    rngs: nnx.Rngs,
  ) -> FrozenDict:
    if self.train_metric_shapes is None:
      _, self.train_metric_shapes = nnx.eval_shape(
        self.train_step,
        optimizer_state=self.optimizer_and_model,
        model_kwargs=batch,
        metrics=None,
        rngs=rngs,
      )
    return jax.tree.map(lambda x: jnp.zeros_like(x), self.train_metric_shapes)

  def __call__(
    self,
    progress_table: ProgressTable,
    train_loader: Iterator,
    epoch_idx: int,
    train_metrics: ImmutableMetrics | None,
    *,
    rngs: nnx.Rngs,
  ) -> tuple[ImmutableMetrics, HostMetrics]:
    """Trains a model for one epoch.

    Args:
        train_loader: Data loader of the training set.
        epoch_idx: Current epoch index.

    Returns:
        A dictionary of the average training metrics over all batches
        for logging.
    """
    # Train model for one epoch, and log avg loss and accuracy
    self.logger.start_epoch(epoch_idx, mode="train")

    for batch in self.tracker(progress_table, train_loader, desc="train epoch"):
      if train_metrics is None:
        train_metrics = self.init_train_metrics(self.batch_to_input_fn(batch), rngs=rngs)
      if self.total_epochs_taken == 0:
        # Log compilation and execution time of the first batch.
        _logger.info("Compiling train_step...")
        start_time = time.time()
        self.optimizer_and_model, train_metrics = self.train_step(
          self.optimizer_and_model,
          self.batch_to_input_fn(batch),
          train_metrics,
          rngs=rngs,
        )
        _logger.info(
          _
          := f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds.",  # noqa: E501
        )
      else:
        # Annotated with step number for TensorBoard profiling.
        with jax.profiler.StepTraceAnnotation(f"train_step_{self.total_epochs_taken}"):
          self.optimizer_and_model, train_metrics = self.train_step(
            self.optimizer_and_model,
            self.batch_to_input_fn(batch),
            train_metrics,
            rngs=rngs,
          )
          if train_metrics is not None and self.enable_progress_bar:
            progress_table.update(
              name="train loss",
              value=train_metrics["loss_step"]["value"],
              aggregate="mean",
              color="blue",
            )
            progress_table.update(
              name="train accuracy",
              value=train_metrics["accuracy_step"]["value"],
              aggregate="mean",
              color="blue",
            )
      for callback in self.train_step_callbacks:
        callback.on_training_step(train_metrics, epoch_idx, self.total_epochs_taken)
      train_metrics = self.logger.log_step(train_metrics)
      self.total_epochs_taken += 1
    train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics)
    return train_metrics, epoch_metrics
