from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
  """Configuration for the Logger class."""

  class_name: str = Field(
    "jax_trainer.logger.Logger", description="Name of the logger class to use."
  )
  log_dir: str = Field(
    default="checkpoints/",
    description="Directory where logs should be saved. If None, a default directory will be created.",
  )

  log_steps_every: int = Field(
    default=50,
    description="Frequency (in steps) at which to log step metrics.",
    gt=0,
  )
  logger_name: str = Field(description="Name of the logger to use.", min_length=1)

  tool: str = Field(
    default="TensorBoard",
    description="Tool to use for logging (e.g. TensorBoard, Wandb).",
    min_length=1,
  )

  project_name: str = Field(
    description="Name of the project to use for logging.",
    min_length=1,
  )
  log_file_verbosity: str = Field(
    default="info", description="Verbosity level for logging to file."
  )

  stderrthreshold: str = Field(
    default="warning", description="Threshold for logging to stderr."
  )

  base_log_dir: str | None = Field(
    default=None,
    description="Base directory for auto-generating log_dir when log_dir is unset.",
  )

  model_log_dir: str | None = Field(
    default=None,
    description="Override the model name component of the auto-generated log path.",
  )
