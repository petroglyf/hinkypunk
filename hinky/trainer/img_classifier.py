import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jax_trainer.logger import LogFreq, LogMetricMode, LogMode, StepMetrics
from jax_trainer.trainer.trainer import TrainerModule


class ImgClassifierTrainer(TrainerModule):
  def loss_function(  # pyrefly: ignore[bad-override]
    self,
    model: nnx.Module,
    batch: dict[str, jax.Array],
    rngs: nnx.Rngs,
    train: bool = True,
  ) -> tuple[jax.Array, StepMetrics]:
    """Cross-entropy loss for image classification.

    Expects batch keys 'input' (images) and 'target' (integer labels).
    """
    imgs = batch["input"]
    labels = batch["target"]
    logits = model(imgs, train=train, rngs=rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    conf_matrix = jnp.zeros((logits.shape[-1], logits.shape[-1]))
    conf_matrix = conf_matrix.at[preds, labels].add(1)
    metrics: StepMetrics = {
      "acc": acc,
      "acc_std": {"value": acc, "mode": LogMetricMode.STD, "log_mode": LogMode.EVAL},
      "acc_max": {
        "value": acc,
        "mode": LogMetricMode.MAX,
        "log_mode": LogMode.TRAIN,
        "log_freq": LogFreq.EPOCH,
      },
      "conf_matrix": {
        "value": conf_matrix,
        "mode": LogMetricMode.SUM,
        "log_mode": LogMode.EVAL,
      },
    }
    return loss, metrics
