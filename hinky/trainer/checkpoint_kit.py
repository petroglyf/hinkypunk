# Standard libraries
import json
import os
from typing import Any, Generic, TypeVar

# JAX/Flax libraries
import jax
import orbax.checkpoint as ocp
from absl import logging
from hinky.optimizer.config import OptimizerConfig

from hinky.trainer.config import CheckpointConfig, ModelConfig, TrainerConfig
from hinky.utils import class_to_name

ModelParamsType = TypeVar("ModelParamsType")


class TrainerModule(Generic[ModelParamsType]):
  trainer_config: TrainerConfig
  optimizer_config: OptimizerConfig
  model_config: ModelConfig
  state: Any


class ModelCheckpoint:
  """Callback to save model parameters and mutable variables to the logging directory."""

  def __init__(
    self,
    params_config: CheckpointConfig,
    trainer: TrainerModule[Any],
  ) -> None:
    """Initializes ModelCheckpoint callback."""
    self.log_dir = trainer.trainer_config.logger.log_dir
    self.trainer = trainer
    self.params_config = params_config

    options = ocp.CheckpointManagerOptions(
      max_to_keep=params_config.save_top_k,
      best_fn=lambda m: m[params_config.monitor],
      best_mode=params_config.mode,
      step_prefix="checkpoint",
      cleanup_tmp_directories=True,
      create=True,
    )
    self.metadata = {
      "trainer": trainer.trainer_config.model_dump(),
      "model": trainer.model_config.model_dump(),
      "optimizer": trainer.optimizer_config.model_dump(),
    }
    self.metadata = jax.tree.map(class_to_name, self.metadata)
    item_handlers = {
      "params": ocp.StandardCheckpointHandler(),
      "metadata": ocp.JsonCheckpointHandler(),
    }
    if params_config.save_mutable_variables:
      item_handlers["mutable_variables"] = ocp.StandardCheckpointHandler()
    if params_config.save_optimizer_state:
      item_handlers["optimizer"] = ocp.StandardCheckpointHandler()
    self.manager = ocp.CheckpointManager(
      directory=os.path.abspath(os.path.join(self.log_dir, "checkpoints/")),
      item_names=tuple(item_handlers.keys()),
      item_handlers=item_handlers,
      options=options,
    )
    self.trainer = trainer

  def on_filtered_validation_epoch_end(
    self, eval_metrics: dict[str, float], epoch_idx: int
  ) -> None:
    self.save_model(self.trainer, eval_metrics, epoch_idx)

  def save_model(
    self, trainer: TrainerModule[Any], eval_metrics: dict[str, float], epoch_idx: int
  ) -> None:
    """Saves model parameters and batch statistics to the logging directory.

    Args:
        eval_metrics: Dictionary of evaluation metrics.
        epoch_idx: Index of the current epoch.
    """
    logging.info(f"Saving model at epoch {epoch_idx} with eval metrics {eval_metrics}.")
    assert self.params_config.monitor in eval_metrics, (
      f'Metric to monitor "{self.params_config.monitor}" not found in eval metrics. Instead has keys: {", ".join(list(eval_metrics.keys()))}'
    )
    save_items = {
      "params": ocp.args.StandardSave(self.trainer.state.params),
      "metadata": ocp.args.JsonSave(self.metadata),
    }
    if trainer.state.mutable_variables is not None:
      save_items["mutable_variables"] = ocp.args.StandardSave(
        trainer.state.mutable_variables
      )
    if self.params_config.save_optimizer_state:
      save_items["optimizer"] = ocp.args.StandardSave(trainer.state.optimizer)
    eval_metrics = {
      k: eval_metrics[k]
      for k in eval_metrics
      if isinstance(eval_metrics[k], (int, float, str, bool))
    }
    save_items = ocp.args.Composite(**save_items)
    self.manager.save(epoch_idx, args=save_items, metrics=eval_metrics)

  def load_model(self, epoch_idx: int=-1) -> dict[str, Any]:
    """Loads model parameters and variables from the logging directory.

    Args:
        epoch_idx: Index of the epoch to load. If -1, loads the best epoch.

    Returns:
        Dictionary of loaded model parameters and additional variables.
    """
    logging.info(f"Loading model at epoch {epoch_idx}.")
    if epoch_idx == -1:
      best = self.manager.best_step()
      if best is None:
        raise ValueError("No checkpoint found.")
      epoch_idx = best
    state_dict = self.manager.restore(epoch_idx)
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    return state_dict

  def finalize(self, status: str | None = None) -> None:
    logging.info("Closing checkpoint manager")
    self.manager.wait_until_finished()
    self.manager.close()


def load_from_checkpoint(checkpoint: str) -> Any:
  """Creates a Trainer object with same hyperparameters and loaded model from a checkpoint directory.

  Args:
      checkpoint: Folder in which the checkpoint and hyperparameter file is stored.

  Returns:
      A Trainer object with model loaded from the checkpoint folder.
  """
  # Load config
  metadata_file = os.path.join(checkpoint, "metadata/metadata")
  assert os.path.isfile(metadata_file), "Could not find metadata file"
  with open(metadata_file, "rb") as f:
    config: dict[str, Any] = json.load(f)
  # Adjust log dir to where its loaded from
  adjusted_checkpoint = checkpoint.split("/")
  if adjusted_checkpoint[-1] == "":
    adjusted_checkpoint = adjusted_checkpoint[:-1]
  if len(adjusted_checkpoint) < 2:
    raise ValueError("Checkpoint path must be at least two levels deep")
  config["trainer"]["logger"]["log_dir"] = os.path.join(*adjusted_checkpoint[:-2])
  raise NotImplementedError(
    "load_from_checkpoint requires a concrete trainer class. "
    "Call this method from a TrainerModule subclass."
  )
