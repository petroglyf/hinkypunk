# pyright: ignore [reportInvalidTypeForm]

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
  [nnx.Module, dict[str, jax.Array], nnx.Rngs, bool], tuple[jax.Array, ImmutableMetrics]
]


class TrainStep:
  def __init__(self, loss_fn: LossFnType, batch_size: int, *, log_grad_norm: bool = False) -> None:
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
      # params_norm = optax.global_norm(state.params)
      # step_metrics["optimizer/params_global_norm"] = {
      #   "value": params_norm,
      #   "log_freq": LogFreq.STEP,
      # }
    metrics = update_metrics(
      metrics,
      step_metrics,
      train=True,
      batch_size=self.batch_size,
    )
    return optimizer_state, metrics
