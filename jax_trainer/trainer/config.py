from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from jax_trainer.callbacks.config import CallbackConfig
from jax_trainer.logger.loggers import LoggerConfig

ModelParamsConfigType = TypeVar("ModelParamsConfigType")


class ModelConfig(BaseModel, Generic[ModelParamsConfigType]):
  name: str = Field(
    description="The full import path of the model class to use.",
    min_length=5,
  )
  hparams: ModelParamsConfigType = Field(
    description="Hyperparameters to initialize the model with.",
  )


class CheckpointConfig(BaseModel):
  monitor: str = Field(description="Metric to monitor for saving the best model.")
  mode: str = Field(
    default="min",
    description='One of {"min", "max"}. In "min" mode, lower metric is better.',
  )
  save_top_k: int = Field(
    ge=1,
    default=1,
    description="Number of best models to save based on the monitored metric.",
  )
  save_optimizer_state: bool = Field(
    default=False,
    description="Whether to save the optimizer state along with the model parameters.",
  )


class TrainerConfig(BaseModel):
  """Configuration for the TrainerModule."""

  name: str = Field(
    description="The full import path of the trainer class to use.",
    min_length=5,
  )

  check_val_every_n_epoch: int = Field(
    default=1,
    ge=0,
    description="Number of epochs between each validation check. If 0, no validation is performed during training.",
  )
  logger: LoggerConfig = Field(description="Configuration for the logger.")

  tabulate_model: bool = Field(
    default=True,
    description="Whether to tabulate the model architecture and save it to a file.",
  )

  tabulate_params: bool = Field(
    default=True,
    description="Whether to tabulate the model parameters and save it to a file.",
  )

  callbacks: list[CallbackConfig] = Field(
    [], description="A list of callback configurations to initialize."
  )
  seed: int = Field(
    default=426,
    description="The random seed to use for the trainer.",
  )
  rngs: list[str] = Field([], description="A list of names for the flax RNGS structure.")

  debug: bool = Field(
    False, description="If True, disables jitting for easier debugging."
  )

  donate_train_state: bool = Field(
    True,
    description="If True, donates the train state to the jitted training step to avoid copies.",
  )
  log_grad_norm: bool = Field(
    False, description="If True, logs the gradient norm during training."
  )

  seed_eval: int = Field(0, description="Random seed for evaluation.")

  detect_nans: bool = Field(False, description="If True, detects NaNs in training.")

  nan_keys: list[str] = Field(
    ["train/loss"], description="Keys to check for NaNs during training."
  )

  enable_progress_bar: bool = Field(
    True, description="If True, enables the progress bar during training and evaluation."
  )

  checkpoint_config: CheckpointConfig | None = Field(
    None, description="Configuration for model checkpointing."
  )

  train_epochs: int = Field(
    default=200,
    ge=1,
    description="Number of epochs to train the model for.",
  )
