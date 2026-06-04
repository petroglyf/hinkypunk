"""TensorBoardLogger implementation for JAX Trainer."""
import logging
from pathlib import Path

import numpy as np
from plotly.graph_objects import Figure
from tensorboard import default, program
from tensorboardX import SummaryWriter

from jax_trainer.logger.config import LoggerConfig
from jax_trainer.logger.metrics import HostMetrics
from jax_trainer.logger.types import LoggerType

_logger = logging.getLogger(__name__)


class TensorBoardLogger(LoggerType):
  """Logger implementation for TensorBoard."""

  def __init__(self, config: LoggerConfig) -> None:
    """Initializes the TensorBoard logger."""
    log_dir = Path(config.log_dir)
    self.log_dir = log_dir
    self.project_name = config.project_name
    self.__start_tensorboard()
    self.writer = SummaryWriter(logdir=str(log_dir))

  def __start_tensorboard(self) -> None:
    """Starts the TensorBoard server."""
    _logger.info("Starting TensorBoard...")
    tensorboard = program.TensorBoard(plugins=default.get_plugins())
    tensorboard.configure(
      argv=[
        "serve",
        "--db=.tensorboard.db",
      ],
    )
    tensorboard.launch()

  def log_metric(
    self,
    metrics_dict: HostMetrics,
    step: int,
  ) -> None:
    for metric_key, metric_value in metrics_dict.items():
      self.writer.add_scalar(metric_key, metric_value, step)

  def finalize(self, status: str) -> None:  # noqa: ARG002
    self.writer.flush()
    self.writer.close()

  def log_image(
    self,
    tag: str,
    image: np.ndarray,
    global_step: int,
    dataformats: str = "CHW",
  ) -> None:
    if dataformats == "HWC":
      image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    self.writer.add_image(tag, image, global_step)

  def log_figure(
    self,
    tag: str,
    figure: Figure,
    global_step: int,
  ) -> None:
    # Convert Plotly figure to a PNG image
    img_bytes = figure.to_image(format="png")
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    self.writer.add_image(tag, img_array, global_step)

  def log_embedding(
    self,
    tag: str,
    mat: np.ndarray,
    metadata: list[str] | None,
    label_img: np.ndarray | None,
    global_step: int,
  ) -> None:
    self.writer.add_embedding(
      mat,
      metadata=metadata,
      label_img=label_img,
      global_step=global_step,
      tag=tag,
    )

  def log_hyperparams(
    self,
    params_dict: dict[str, str | int | float],
  ) -> None:
    self.writer.add_hparams(params_dict, {})
