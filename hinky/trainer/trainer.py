"""Trainer module for training and evaluating JAX/Flax models."""

import logging
import os
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import yaml
from flax import nnx
from progress_table.progress_table import ProgressTable, TableProgressBar
from pyarrow import Table
from pydantic import create_model

from hinky.datasets import DatasetModule
from hinky.datasets.data_struct import FullDatasetSpecification
from hinky.logger import ImmutableMetrics
from hinky.logger.config import LoggerConfig
from hinky.optimizer.config import OptimizerConfig
from hinky.optimizer.optimizer_constructor import OptimizerBuilder
from hinky.trainer.checkpoint_kit import ModelCheckpoint
from hinky.trainer.config import ModelConfig, TrainerConfig
from hinky.utils import resolve_import

ModelParamsType = TypeVar("ModelParamsType")


class TrainerModule(Generic[ModelParamsType]):
  def __init__(
    self,
    trainer_config: TrainerConfig,
    model_config: ModelConfig[ModelParamsType],
    optimizer_config: OptimizerConfig,
    dataset_config: FullDatasetSpecification,
    dataset: DatasetModule,
  ) -> None:
    """A basic Trainer module for logging, model initialization, training loop, and callbacks.

    Args:
        trainer_config: A dictionary containing the trainer configuration.
        model_config: A dictionary containing the model configuration.
        optimizer_config: A dictionary containing the optimizer configuration.
        dataset: The dataset module containing the training, validation and test sets.
        dataset_config: A dictionary containing the dataset configuration.
    """
    super().__init__()
    self.trainer_config = trainer_config
    self.model_config = model_config
    self.optimizer_config = optimizer_config
    self.dataset = dataset
    self.dataset_config = dataset_config

    # Default properties for trainer config
    self.trainer_config.check_val_every_n_epoch = self.trainer_config.check_val_every_n_epoch
    self.prepare_rngs()

    # Create empty model. Note: no parameters yet

    self.build_model(model_config)
    # Init trainer parts
    # self.create_jitted_functions()  # noqa: ERA001
    self.init_logger(self.trainer_config.logger)
    self.init_callbacks()
    self.checkpoint_manager = None
    if trainer_config.checkpoint_config is not None:
      self.checkpoint_manager = ModelCheckpoint(
        params_config=trainer_config.checkpoint_config,
        # pyrefly: ignore [bad-argument-type]
        trainer=self,
      )

  def batch_to_input(self, batch: Table) -> dict[str, Any]:
    raise NotImplementedError

  def prepare_rngs(self) -> None:
    root_rng_train = jax.random.key(self.trainer_config.seed)
    root_rng_eval = jax.random.key(self.trainer_config.seed_eval)
    n_rngs = len(self.trainer_config.rngs)

    # Initialize the rngs for the training pass
    rngs_seeds_train = jax.random.split(root_rng_train, n_rngs + 1)
    rngs_params_train = dict(
      zip(["params", *self.trainer_config.rngs], rngs_seeds_train, strict=False),
    )
    self.rngs_train = nnx.Rngs(**rngs_params_train)

    # Initialize the rngs for the evaluation pass
    rngs_seeds_eval = jax.random.split(root_rng_eval, n_rngs + 1)
    rngs_params_eval = dict(
      zip(["params", *self.trainer_config.rngs], rngs_seeds_eval, strict=False),
    )
    self.rngs_eval = nnx.Rngs(**rngs_params_eval)

  def build_model(self, model_config: ModelConfig[ModelParamsType]) -> None:
    """Creates the model class from the model_config.

    Args:
        model_config: A dictionary containing the model configuration.
    """
    # Create model
    model_class = resolve_import(model_config.name)
    hparams = model_config.hparams
    self.model = model_class(**hparams, rngs=self.rngs_train)

  def init_logger(self, logger_config: LoggerConfig) -> None:
    """Initializes a logger and creates a logging directory.

    Args:
        logger_config: A dictionary containing the specification of the logger.
    """
    all_config_type = create_model(
      "FullModelConfig",
      trainer=TrainerConfig,
      model=ModelConfig[ModelParamsType],
      optimizer=OptimizerConfig,
    )
    all_config = all_config_type(
      trainer=self.trainer_config,
      model=self.model_config,
      optimizer=self.optimizer_config,
    )
    logger_class = resolve_import(logger_config.class_name)
    self.logger = logger_class(logger_config, self.model_config.name)
    # Log the hyperparams
    # hparams = flatten_configdict(all_config)
    # hparams = jax.tree_map(class_to_name, hparams)
    # logger.log_hyperparams(hparams)
    return

    # Save config and exmp_input
    log_dir = self.logger.log_dir
    logging.info(f"Logging at {log_dir}")
    self.log_dir = log_dir
    self.trainer_config.logger.log_dir = log_dir
    os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)

    if not os.path.isfile(os.path.join(log_dir, "config.yaml")):
      yaml_str = yaml.dump(all_config.model_dump())
      with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        f.write(yaml_str)

  def init_callbacks(self):
    """Initializes the callbacks defined in the trainer config."""
    self.callbacks = []
    self.train_step_callbacks = []
    callback_configs = self.trainer_config.callbacks

    for callback_config in callback_configs:
      logging.info(f"Initializing callback {callback_config.name}")
      # if callback_config.get("class_name", None) is not None:
      callback_class = resolve_import(callback_config.class_name)
      callback_class_config = callback_class.encapsulate_config(callback_config.options)
      callback = callback_class(
        params_config=callback_class_config, callback_config=callback_config, trainer=self
      )
      self.callbacks.append(callback)
      if hasattr(callback, "on_training_step"):
        self.train_step_callbacks.append(callback)

  def set_dataset(self, dataset: DatasetModule) -> None:
    for callback in self.callbacks:
      callback.set_dataset(dataset)
    self.dataset = dataset

  def init_optimizer(self, num_epochs: int, num_train_steps_per_epoch: int) -> None:
    """Initializes the optimizer and learning rate scheduler.

    Args:
        num_epochs: Number of epochs the model will be trained for.
        num_train_steps_per_epoch: Number of training steps per epoch.
    """
    builder = OptimizerBuilder(self.optimizer_config)
    optimizer, lr_schedule = builder.build_optimizer(
      num_epochs=num_epochs,
      num_train_steps_per_epoch=num_train_steps_per_epoch,
    )
    self.lr_schedule = lr_schedule  # Save for logging
    # Initialize training state
    self.state = nnx.ModelAndOptimizer(self.model, optimizer, wrt=nnx.Param)

  def create_jitted_functions(self) -> None:
    """Creates jitted versions of the training and evaluation functions.

    If self.debug is True, not jitting is applied.
    """
    # pyrefly: ignore [missing-attribute]
    train_step, eval_step = self.create_functions()
    if self.trainer_config.debug:  # Skip jitting
      logging.info("Skipping jitting due to debug=True")
      self.train_step = train_step
      self.eval_step = eval_step
    else:  # Jit
      logging.info("Jitting train_step and eval_step...")
      train_donate_argnames = ["metrics"]  # Donate metrics to avoid copying.
      if self.trainer_config.donate_train_state:
        train_donate_argnames.append("optimizer_state")
      self.train_step = nnx.jit(
        train_step,
        # donate_argnames=["metrics"],  # noqa: ERA001
      )
      self.eval_step = nnx.jit(
        eval_step,
        # donate_argnames=["metrics"],  # Donate metrics to avoid copying.  # noqa: ERA001
      )

  def loss_function(
    self,
    model: nnx.Module,
    batch: dict[str, jax.Array],
    rngs: nnx.Rngs,
    train: bool = True,
  ) -> tuple[jax.Array, ImmutableMetrics]:
    """The loss function that is used for training.

    This function needs to be overwritten by a subclass.
    """
    raise NotImplementedError

  def tracker(
    self, progress_table: ProgressTable, iterator: Iterator, desc: str,  # noqa: ARG002
  ) -> Iterator | TableProgressBar:
    """Wraps an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

    Args:
        iterator: Iterator to wrap in tqdm.
        kwargs: Additional arguments to tqdm.

    Returns:
        Wrapped iterator if progress bar is enabled, otherwise same iterator
        as input.
    """
    if self.trainer_config.enable_progress_bar:
      return progress_table(
        iterable=iterator,
        show_throughput=False,
        show_eta=True,
      )
    return iterator

  def on_training_start(self) -> None:
    """Method called before training is started.

    Can be used for additional initialization operations etc.
    """
    logging.info("Starting training")
    for callback in self.callbacks:
      callback.on_training_start()

  def on_training_end(self) -> None:
    """Method called after training has finished.

    Can be used for additional logging or similar.
    """
    logging.info("Finished training")
    for callback in self.callbacks:
      callback.on_training_end()

  def continue_with_batch(self) -> bool: # pyrefly: ignore [bad-return]
    """Determines if we're done with the batch."""

  def on_train_step_start(self, model: nnx.Module, batch_data: dict[str, jax.Array], rngs: nnx.Rngs) -> dict[str, jnp.ndarray] | None:
    """It is possible to use multiple passes on a batch to generate more data.

    Use this function to translate batch input to something else useful.
    """
    return batch_data

  def on_train_step_end(self, model: nnx.Module, batch_data: dict[str, jax.Array], rngs: nnx.Rngs) -> dict[str, jnp.ndarray] | None:
    """It is possible to use multiple passes on a batch to generate more data.

    Use this function to translate batch input to something else useful.
    """
    return None

  def on_training_epoch_start(self, epoch_idx: int) -> None:
    """Method called at the start of each training epoch. Can be used for additional logging or similar.

    Args:
        epoch_idx: Index of the training epoch that has started.
    """
    logging.info(f"Starting training epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_training_epoch_start(epoch_idx)

  def on_training_epoch_end(self, train_metrics: dict[str, Any], epoch_idx: int) -> None:
    """Method called at the end of each training epoch. Can be used for additional logging or similar.

    Args:
        epoch_idx: Index of the training epoch that has finished.
    """
    logging.info(f"Finished training epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_training_epoch_end(train_metrics, epoch_idx)

  def on_validation_epoch_start(self, epoch_idx: int) -> None:
    """Method called at the start of each validation epoch. Can be used for additional logging or similar.

    Args:
        epoch_idx: Index of the training epoch at which validation was started.
    """
    logging.info(f"Starting validation epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_validation_epoch_start(epoch_idx)

  def on_validation_epoch_end(self, eval_metrics: dict[str, Any], epoch_idx: int) -> None:
    """Method called at the end of each validation epoch. Can be used for additional logging and evaluation.

    Args:
        epoch_idx: Index of the training epoch at which validation was performed.
        eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
        val_loader: Data loader of the validation set, to support additional
            evaluation.
    """
    logging.info(f"Finished validation epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_validation_epoch_end(eval_metrics, epoch_idx)
    if (
      self.checkpoint_manager is not None
      and self.trainer_config.checkpoint_config
      and epoch_idx % self.trainer_config.checkpoint_config.every_n_epochs == 0
    ):
      self.checkpoint_manager.save_model(eval_metrics, epoch_idx)

  def on_test_epoch_start(self, epoch_idx: int) -> None:
    """Method called at the start of each test epoch. Can be used for additional logging or similar.

    Args:
        epoch_idx: Index of the training epoch at which testing was started.
    """
    logging.info(f"Starting test epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_test_epoch_start(epoch_idx)

  def on_test_epoch_end(self, test_metrics: dict[str, Any], epoch_idx: int) -> None:
    """Method called at the end of each test epoch.

    Can be used for additional logging and evaluation.

    Args:
        epoch_idx: Index of the training epoch at which testing was performed.
        test_metrics: A dictionary of the test metrics. New metrics added to
            this dictionary will be logged as well.
        test_loader: Data loader of the test set, to support additional
            evaluation.
    """
    logging.info(f"Finished test epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_test_epoch_end(test_metrics, epoch_idx)

  def load_model(self, epoch_idx: int = -1, raise_if_not_found: bool = True) -> None:
    """Loads model parameters and batch statistics from the logging directory."""
    logging.info(f"Loading model from epoch {epoch_idx}")
    state_dict = None
    for callback in self.callbacks:
      if isinstance(callback, ModelCheckpoint):
        state_dict = callback.load_model(epoch_idx)
        break
    if state_dict is None:
      if raise_if_not_found:
        raise ValueError("No model checkpoint callback found in callbacks.")
      logging.warning("No model checkpoint callback found in callbacks.")
    else:
      self.restore(state_dict)

  def restore(self, state_dict: dict[str, Any]) -> None:
    """Restores the state of the trainer from a state dictionary.

    Args:
        state_dict: State dictionary to restore from.
    """
    logging.info("Restoring trainer state with keys " + str(state_dict.keys()))
    state_dict.pop("metrics")
    state_dict.pop("metadata")

