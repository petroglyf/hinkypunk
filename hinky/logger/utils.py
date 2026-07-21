import importlib.util
import os

from hinky.logger.config import LoggerConfig
from hinky.logger.types import LoggerType


def get_logging_dir(logger_config: LoggerConfig, model_name: str) -> tuple[str, str | None]:
  """Returns the logging directory and version.

  Args:
      logger_config (LoggerConfig): The logger config.
      model_name (str): The name of the model. Used for getting the model name for the default logging directory.

  Returns:
      Tuple[str, str]: The logging directory and version.
  """
  # Determine logging directory
  log_dir = logger_config.log_dir
  if log_dir == "None":
    log_dir = None
  if not log_dir:
    base_log_dir = logger_config.base_log_dir
    if base_log_dir is None:
      raise ValueError(_ := "LoggerConfig.base_log_dir must be set when log_dir is not provided.")
    # Prepare logging
    model_name = model_name.split(".")[-1] if logger_config.model_log_dir is None else logger_config.model_log_dir
    log_dir = os.path.join(base_log_dir, model_name)
    if logger_config.logger_name is not None:
      log_dir = os.path.join(log_dir, logger_config.logger_name)
      version = ""
    else:
      version = None
  else:
    version = ""
  return log_dir, version


def build_tool_logger(logger_config: LoggerConfig, model_name: str) -> LoggerType:
  """Builds the logger tool object, either Tensorboard or Weights and Biases.

  Args:
      logger_config (LoggerConfig): The logger config.
      model_name (str): The name of the model.

  Returns:
      The logger tool object.
  """
  # Create logger object
  logger_type = logger_config.tool.lower()
  if logger_type == "tensorboard":
    if importlib.util.find_spec("tensorboard") is None:
      msg = "Tensorboard is not installed but LoggerConfig.tool='TensorBoard' was requested."
      raise ImportError(msg)
    tensorboard_logger = importlib.import_module("hinky.logger.bundled.TensorBoardLogger")
    logger = tensorboard_logger.TensorBoardLogger(logger_config)
  elif logger_type == "wandb":
    if importlib.util.find_spec("wandb") is None:
      msg = "WandB is not installed but LoggerConfig.tool='wandb' was requested."
      raise ImportError(msg)
    wandb_logger = importlib.import_module("hinky.logger.bundled.WandbLogger")
    logger = wandb_logger.WandbLogger(logger_config)
  else:
    raise ValueError(_ := f"Unknown logger type {logger_type}.")
  return logger
