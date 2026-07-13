"""Base class for callbacks."""

from typing import Any, Generic, TypeVar

from jax_trainer.callbacks.config import CallbackConfig

OptionsConfigtype = TypeVar("OptionsConfigtype", bound=CallbackConfig)


class Callback(Generic[OptionsConfigtype]):
  """Base class for callbacks."""

  def __init__(self, config: CallbackConfig) -> None:
    """Base class for callbacks.

    Args:
        config: Configuration dictionary.
        trainer: Trainer object.
    """
    self.__callback_config = config

  @classmethod
  def encapsulate_config(cls, options: dict[str, Any]) -> OptionsConfigtype: # pyrefly: ignore[bad-return]
    """Translates a dictionary of options into a pydantic model."""

  def on_training_step(
    self,
    train_metrics: Any,
    epoch_idx: int,
    global_step: int,
  ) -> None:
    """Called after each training step."""

  def on_training_start(self) -> None:
    """Called at the beginning of training."""

  def on_training_end(self) -> None:
    """Called at the end of training."""

  def on_training_epoch_start(self, epoch_idx: int) -> None:
    """Called at the beginning of each training epoch.

    Args:
        train_state: The current training state.
        epoch_idx: Index of the current epoch.
    """
    if (
      self.__callback_config.every_n_epochs is not None
      and epoch_idx % self.__callback_config.every_n_epochs != 0
    ):
      return
    self.on_filtered_training_epoch_start(epoch_idx)

  def on_filtered_training_epoch_start(self, epoch_idx: int) -> None:
    """Called at the beginning of each `every_n_epochs` training epoch.

    To be implemented by subclasses.

    Args:
        train_state: The current training state.
        epoch_idx: Index of the current epoch.
    """

  def on_training_epoch_end(
    self,
    train_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Called at the end of each training epoch.

    Args:
        train_state: The current training state.
        train_metrics: Dictionary of training metrics of the current epoch.
        epoch_idx: Index of the current epoch.
    """
    if (
      self.__callback_config.every_n_epochs is not None
      and epoch_idx % self.__callback_config.every_n_epochs == 0
    ):
      self.on_filtered_training_epoch_end(train_metrics, epoch_idx)

  def on_filtered_training_epoch_end(
    self,
    train_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Called at the end of each `every_n_epochs` training epoch.

    To be implemented by subclasses.

    Args:
        train_state: The current training state.
        train_metrics: Dictionary of training metrics of the current epoch.
        epoch_idx: Index of the current epoch.
    """

  def on_validation_epoch_start(
    self,
    epoch_idx: int,
  ) -> None:
    """Called at the beginning of validation."""
    if (
      self.__callback_config.every_n_epochs is not None
      and epoch_idx % self.__callback_config.every_n_epochs == 0
    ):
      self.on_filtered_validation_epoch_start(epoch_idx)

  def on_filtered_validation_epoch_start(
    self,
    epoch_idx: int,
  ) -> None:
    """Called at the beginning of `every_n_epochs` validation. To be implemented by subclasses.

    Args:
        train_state: The current training state.
        epoch_idx: Index of the current epoch.
    """

  def on_validation_epoch_end(
    self,
    eval_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Called at the end of each validation epoch.

    Args:
        train_state: The current training state.
        eval_metrics: Dictionary of evaluation metrics of the current epoch.
        epoch_idx: Index of the current epoch.
    """
    if (
      self.__callback_config.every_n_epochs is not None
      and epoch_idx % self.__callback_config.every_n_epochs != 0
    ):
      return
    self.on_filtered_validation_epoch_end(eval_metrics, epoch_idx)

  def on_filtered_validation_epoch_end(
    self,
    eval_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Called at the end of each `every_n_epochs` validation epoch.

    To be implemented bysubclasses.

    Args:
        train_state: The current training state.
        eval_metrics: Dictionary of evaluation metrics of the current epoch.
        epoch_idx: Index of the current epoch.
    """

  def on_test_epoch_start(
    self,
    epoch_idx: int,
  ) -> None:
    """Called at the beginning of testing.

    To be implemented by subclasses.
    """

  def on_test_epoch_end(
    self,
    test_metrics: dict[str, float],
    epoch_idx: int,
  ) -> None:
    """Called at the end of each test epoch. To be implemented by subclasses.

    Args:
        train_state: The current training state.
        test_metrics: Dictionary of test metrics of the current epoch.
        epoch_idx: Index of the current epoch.
    """

  def finalize(self, status: str | None = None) -> None:
    """Called at the end of the whole training process.

    To be implemented by subclasses.
    """
