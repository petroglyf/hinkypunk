# pyright: ignore [reportInvalidTypeForm]

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, Final

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import FrozenDict

from jax_trainer.logger import (
  ImmutableMetrics,
  update_metrics,
)

from .trainer import TrainerModule

_logger = logging.getLogger(__name__)
LossFnType = Callable[
  [nnx.Module, dict[str, jax.Array], nnx.Rngs, bool], tuple[jax.Array, ImmutableMetrics]
]


class EvalStep:
  def __init__(
    self,
    loss_fn: LossFnType,
    batch_size: int,
  ) -> None:
    self.loss_function = loss_fn
    self.batch_size: Final[int] = batch_size
    self.eval_metric_shapes = None

  def __call__(
    self,
    optimizer_and_model: nnx.ModelAndOptimizer,
    model_kwargs: dict[str, jax.Array],
    metrics: ImmutableMetrics | None,
    *,
    rngs: nnx.Rngs,
  ) -> ImmutableMetrics:
    all_value_and_grad_out = nnx.value_and_grad(self.loss_function, has_aux=True)(
      optimizer_and_model.model,
      model_kwargs,
      rngs=rngs,
      train=False,
    )
    (_, step_metrics), _ = all_value_and_grad_out
    metrics = update_metrics(
      metrics,
      step_metrics,
      train=False,
      batch_size=self.batch_size,
    )

    return metrics

  def init_eval_metrics(
    self,
    optimizer_and_model: nnx.ModelAndOptimizer,
    batch: dict[str, Any] | None = None,
    *,
    rngs: nnx.Rngs,
  ) -> FrozenDict:
    if self.eval_metric_shapes is None:
      self.eval_metric_shapes = nnx.eval_shape(
        self,
        optimizer_and_model=optimizer_and_model,
        model_kwargs=batch,
        metrics=None,
        rngs=rngs,
      )
    return jax.tree.map(lambda x: jnp.zeros_like(x), self.eval_metric_shapes)

  def test_eval_function(
    self,
    trainer: TrainerModule[Any],
    val_loader: Iterator,
    *,
    rngs: nnx.Rngs,
  ) -> None:
    """Tests the evaluation function on a single batch.

    This is useful to check if the functions have the correct signature and return the correct
    values. This prevents annoying errors that occur at the first evaluation step.

    This function does not test the training function anymore. This is because the training
    function is already executed in the first epoch and we change its jit signature to donate
    the train state and metrics. Thus, executing a training step requires updating the train
    state, which we would not want to do here. The compilation time is logged during the very
    first training step.

    Args:
        val_loader: Data loader of the validation set.
    """
    _logger.info("Verifying evaluation function...")
    val_batch = trainer.batch_to_input(next(val_loader))
    eval_metrics = self.init_eval_metrics(
      optimizer_and_model=trainer.state,
      batch=val_batch,
      rngs=rngs,
    )
    start_time = time.time()
    _logger.info("Testing and compiling eval_step...")
    _ = self(
      optimizer_and_model=trainer.state,
      model_kwargs=val_batch,
      metrics=eval_metrics,
      rngs=rngs,
    )
    _logger.info(_ := f"Successfully completed in {time.time() - start_time:.2f} seconds.")
