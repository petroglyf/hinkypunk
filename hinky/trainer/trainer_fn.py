from hinky.datasets.data_struct import PermissibleArrowTables
import logging
from collections.abc import Callable, Iterator
from itertools import chain
from typing import Any

import jax
import numpy as np
from flax import nnx
from progress_table.progress_table import ProgressTable

from hinky.datasets import DatasetModule
from hinky.logger import (
  HostMetrics,
  ImmutableMetrics,
)

from .episode_stepper import EpisodeProcessAsStep
from .eval_stepper import EvalStep
from .train_stepper import TrainStep
from .trainer import TrainerModule

_logger = logging.getLogger(__name__)


def _peek(iterable: Iterator[Any]) -> tuple[Any, Iterator] | None:
  try:
    first = next(iterable)
  except StopIteration:
    return None
  return first, chain([first], iterable)


def _eval_model(
  eval_stepper: EvalStep,
  callable_step: Callable[
    [nnx.ModelAndOptimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
    ImmutableMetrics,
  ],
  trainer: TrainerModule[Any],
  val_loader: Iterator,
  mode: str,
  epoch_idx: int,
  *,
  rngs: nnx.Rngs,
) -> HostMetrics:
  """Evaluates the model on a dataset.

  Args:
      data_loader: Data loader of the dataset to evaluate on.
      mode: Whether 'val' or 'test'
      epoch_idx: Current epoch index.

  Returns:
      A dictionary of the evaluation metrics, averaged over data points
      in the dataset.
  """
  # Test model on all images of a data loader and return avg loss
  trainer.logger.start_epoch(epoch_idx, mode=mode)
  peek_value = _peek(val_loader)
  assert peek_value is not None
  example_input, val_loader = peek_value
  eval_metrics = eval_stepper.init_eval_metrics(
    optimizer_and_model=trainer.state,
    batch=trainer.batch_to_input(example_input),
    rngs=rngs,
  )
  step_count = 0
  progress_table = ProgressTable(
    pbar_embedded=False,  # Do not use embedded pbar
    pbar_style="angled alt red blue",
  )
  for batch in trainer.tracker(progress_table, val_loader, desc=mode.capitalize()):
    eval_metrics = callable_step(
      trainer.state,
      trainer.batch_to_input(batch),
      eval_metrics,
      rngs,
    )
    step_count += 1
  if step_count == 0:
    _logger.warning(_ := f"No batches in {mode} loader at epoch {epoch_idx}.")
  _, metrics = trainer.logger.end_epoch(eval_metrics, save_metrics=True)
  return metrics


def _get_step_fns(
  trainer: TrainerModule[Any],
  compile: bool,
) -> tuple[
  TrainStep,
  Callable[
    [nnx.ModelAndOptimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
    ImmutableMetrics,
  ],
  EvalStep,
]:
  train_step = TrainStep(
    trainer.loss_function,
    trainer.dataset_config.training_params.batch_size,
    log_grad_norm=trainer.trainer_config.log_grad_norm,
  )
  eval_step = EvalStep(trainer.loss_function, trainer.dataset_config.training_params.batch_size)
  if compile:
    _logger.info("Jitting train_step and eval_step...")
    train_donate_argnames = ["metrics"]  # Donate metrics to avoid copying.
    if trainer.trainer_config.donate_train_state:
      train_donate_argnames.append("optimizer_state")
    # pyrefly: ignore [bad-return]
    return nnx.jit(train_step), nnx.jit(eval_step), eval_step
  _logger.info("Skipping jitting due to debug=True")
  # pyrefly: ignore [bad-return]
  return train_step, eval_step, eval_step


def train_model(
  trainer: TrainerModule[Any],
  ds_tables: DatasetModule[Any, PermissibleArrowTables],
  num_epochs: int = 500,
) -> dict[str, Any]:
  """Starts a training loop for the given number of epochs.

  Args:
      train_loader: Data loader of the training set.
      val_loader: Data loader of the validation set.
      test_loader: If given, best model will be evaluated on the test set.
      num_epochs: Number of epochs for which to train the model.

  Returns:
      A dictionary of the train, validation and evt. test metrics for the
      best model on the validation set.
  """
  # Create optimizer and the scheduler for the given number of epochs
  trainer.init_optimizer(num_epochs, len(ds_tables.train))
  train_step_call, eval_step_call, eval_step = _get_step_fns(
    trainer, not trainer.trainer_config.debug
  )
  epoch_trainer = EpisodeProcessAsStep(
    trainer.tracker, # pyrefly: ignore [bad-argument-type]
    trainer.logger,
    trainer.train_step_callbacks,
    trainer.batch_to_input,
    trainer.on_train_step_start, # pyrefly: ignore [bad-argument-type]
    trainer.continue_with_batch,
    trainer.on_train_step_end, # pyrefly: ignore [bad-argument-type]
    trainer.state,
    train_step_call,
    enable_progress_bar=trainer.trainer_config.enable_progress_bar,
  )
  # Prepare training loop
  trainer.on_training_start()
  if ds_tables.val:
    val_tbl = ds_tables.val
    # val_ds.set_format("numpy")
    # val_ds = val_ds.cast_column("image", Image(decode=True))
    eval_step.test_eval_function(trainer, val_tbl, rngs=trainer.rngs_eval)
  all_eval_metrics = {}
  train_metrics = None
  training_failed = False
  progress_table = ProgressTable(
    pbar_embedded=False,  # Do not use embedded pbar
    pbar_style="angled alt red blue",
  )
  for epoch_idx in trainer.tracker(progress_table, iter(range(1, num_epochs + 1)), desc=""):
    progress_table["epoch"] = epoch_idx
    trainer.on_training_epoch_start(epoch_idx)
    train_ds = Dataset(ds_tables.train)
    train_ds.set_format("numpy")
    train_metrics, epoch_metrics = epoch_trainer(
      progress_table,
      train_ds.to_batches(max_chunksize=trainer.dataset_config.training_params.batch_size),
      epoch_idx=epoch_idx,
      train_metrics=train_metrics,
      rngs=trainer.rngs_train,
    )
    if trainer.trainer_config.detect_nans:
      nan_keys = trainer.trainer_config.nan_keys
      if isinstance(nan_keys, str):
        nan_keys = (nan_keys,)
      if any(np.isnan(epoch_metrics.get(key, 0.0)).any() for key in nan_keys):
        _logger.error(
          _ := f"NaN detected in epoch metrics of epoch {epoch_idx}. Aborting training.",
        )
        training_failed = True
        break
    trainer.on_training_epoch_end(epoch_metrics, epoch_idx)
    # Validation every N epochs
    run_validation = (
      ds_tables.val is not None
      and trainer.trainer_config.check_val_every_n_epoch > 0
      and epoch_idx % trainer.trainer_config.check_val_every_n_epoch == 0
    )
    if run_validation and ds_tables.val is not None:
      trainer.on_validation_epoch_start(epoch_idx)
      eval_metrics = _eval_model(
        eval_stepper=eval_step,
        callable_step=eval_step_call,
        trainer=trainer,
        val_loader=iter(ds_tables.val),
        mode="val",
        epoch_idx=epoch_idx,
        rngs=trainer.rngs_eval,
      )
      all_eval_metrics[epoch_idx] = eval_metrics
      trainer.on_validation_epoch_end(eval_metrics, epoch_idx)
      loss = eval_metrics["val/loss"]
      if trainer.trainer_config.enable_progress_bar:
        progress_table.update(
          name="valid loss",
          value=loss,
          color="red",
        )
        progress_table.update(
          name="valid accuracy",
          value=eval_metrics["val/accuracy"],
          color="red bold",
        )

    progress_table.next_row(split=run_validation)
  progress_table.close()
  if not training_failed:
    trainer.on_training_end()
    # Test best model if possible
    # if datasets.test is not None:
    #   trainer.on_test_epoch_start(epoch_idx)
    #   test_metrics = self.eval_model(datasets.test, mode="test", epoch_idx=epoch_idx)
    #   trainer.on_test_epoch_end(test_metrics, epoch_idx)
    #   all_eval_metrics["test"] = test_metrics
  # Close logger
  trainer.logger.finalize("success" if not training_failed else "failed")
  for callback in trainer.callbacks:
    callback.finalize("success" if not training_failed else "failed")
  if trainer.checkpoint_manager is not None:
    trainer.checkpoint_manager.finalize("success" if not training_failed else "failed")
  return all_eval_metrics
