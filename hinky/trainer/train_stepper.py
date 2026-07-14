"""Single JAX training step: forward/backward pass, optimizer update, and metrics.

Wraps a loss function into a callable that computes gradients via
nnx.value_and_grad, applies them in place to an nnx.ModelAndOptimizer, and
merges the resulting step metrics (optionally including gradient norms) into
the running metrics dict via jax_trainer.logger.update_metrics.
"""

from collections.abc import Callable
from typing import Final

# JAX/Flax libraries
import jax
import optax
from flax import nnx

from jax_trainer.logger import (
  ImmutableMetrics,
  LogFreq,
  LogMetricMode,
  update_metrics,
)

LossFnType = Callable[
  [nnx.Module, dict[str, jax.Array], nnx.Rngs, bool], tuple[jax.Array, ImmutableMetrics],
]


class TrainStep:
  """Callable that performs one gradient-update step for a training batch."""

  def __init__(self, loss_fn: LossFnType, batch_size: int, *, log_grad_norm: bool = False) -> None:
    """Configures the step function with a loss function and metric options.

    Args:
      loss_fn: Loss function taking (model, batch, rngs, train) and returning
        (loss, metrics), where metrics follows the ImmutableMetrics structure.
      batch_size: Number of examples per batch, used to weight metric aggregation.
      log_grad_norm: Whether to additionally log the global gradient norm
        (both the per-step value and its running max over the epoch).
    """
    self.loss_function = loss_fn
    self.log_grad_norm: Final[bool] = log_grad_norm
    self.batch_size: Final[int] = batch_size

  def __call__(
    self,
    optimizer_state: nnx.ModelAndOptimizer,
    model_kwargs: dict[str, jax.Array],
    metrics: ImmutableMetrics | None,
    *,
    rngs: nnx.Rngs,
  ) -> tuple[nnx.ModelAndOptimizer, ImmutableMetrics]:
    all_value_and_grad_out = nnx.value_and_grad(self.loss_function, has_aux=True)(
      model=optimizer_state.model,
      batch=model_kwargs,
      rngs=rngs,
      train=True,
    )
    (_, step_metrics), grads = all_value_and_grad_out
    # In place updates. -- this usualy takes a model parameter but the old version is just grads.
    optimizer_state.update(grads)

    if self.log_grad_norm:
      grad_norm = optax.global_norm(grads)
      step_metrics["optimizer/grad_global_norm"] = {
        "value": grad_norm,
        "log_freq": LogFreq.STEP,
      }
      step_metrics["optimizer/grad_global_norm_max"] = {
        "value": grad_norm,
        "mode": LogMetricMode.MAX,
        "log_freq": LogFreq.EPOCH,
      }
    metrics = update_metrics(
      metrics,
      step_metrics,
      train=True,
      batch_size=self.batch_size,
    )
    return optimizer_state, metrics
