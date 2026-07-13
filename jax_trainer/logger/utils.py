import os

from jax_trainer.logger.bundled.TensorBoardLogger import TensorBoardLogger
from jax_trainer.logger.config import LoggerConfig


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
      raise ValueError("LoggerConfig.base_log_dir must be set when log_dir is not provided.")
    # Prepare logging
    if logger_config.model_log_dir is None:
      model_name = model_name.split(".")[-1]
    else:
      model_name = logger_config.model_log_dir
    log_dir = os.path.join(base_log_dir, model_name)
    if logger_config.logger_name is not None:
      log_dir = os.path.join(log_dir, logger_config.logger_name)
      version = ""
    else:
      version = None
  else:
    version = ""
  return log_dir, version


def build_tool_logger(logger_config: LoggerConfig, model_name: str) -> TensorBoardLogger:
  """Builds the logger tool object, either Tensorboard or Weights and Biases.

  Args:
      logger_config (LoggerConfig): The logger config.
      model_name (str): The name of the model.

  Returns:
      The logger tool object.
  """
  # Determine logging directory
  log_dir, version = get_logging_dir(logger_config, model_name)
  # Create logger object
  logger_type = logger_config.tool.lower()
  if logger_type == "tensorboard":
    logger = TensorBoardLogger(logger_config)

  # elif logger_type == "wandb":
  #   if version is None:
  #     version = time.strftime("%Y%m%d-%H%M%S")
  #     # Add random string to make sure the version is unique
  #     config_string = str(full_config.to_dict())
  #     version += "-" + str(abs(hash(config_string)) % (10**12))
  #   dict_config = full_config.to_dict()
  #   dict_config["checkpoint_log_dir"] = log_dir
  #   dict_config["checkpoint_version"] = version
  #   logger = WandbLogger(
  #     name=logger_config.get("wandb_name", None),
  #     project=logger_config.get("project_name", None),
  #     save_dir=log_dir,
  #     version=version,
  #     config=dict_config,
  #     log_model=False,
  #   )
  else:
    raise ValueError(f"Unknown logger type {logger_type}.")
  return logger
