"""WandbLogger implementation for JAX Trainer."""
import io
import logging
from pathlib import Path

import altair as alt
import numpy as np
import wandb
from PIL import Image

from hinky.logger.config import LoggerConfig
from hinky.logger.metrics import HostMetrics
from hinky.logger.types import LoggerType

_logger = logging.getLogger(__name__)


class WandbLogger(LoggerType):
  """Logger implementation for Weights & Biases.

  Reads the API key from the WANDB_API_KEY environment variable (the wandb
  default) unless wandb is otherwise configured (e.g. via `wandb login`).
  """

  def __init__(self, config: LoggerConfig) -> None:
    """Initializes the W&B run."""
    self.log_dir = Path(config.log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.project_name = config.project_name
    self.run = wandb.init(
      project=config.project_name,
      name=config.logger_name,
      dir=str(self.log_dir),
    )

  def log_metric(
    self,
    metrics_dict: HostMetrics,
    step: int,
  ) -> None:
    self.run.log(dict(metrics_dict), step=step)

  def finalize(self, status: str) -> None:
    self.run.summary["status"] = status
    exit_code = 0 if status == "success" else 1
    self.run.finish(exit_code=exit_code)

  def log_image(
    self,
    tag: str,
    image: np.ndarray,
    global_step: int,
    dataformats: str = "CHW",
  ) -> None:
    if dataformats == "CHW":
      image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    self.run.log({tag: wandb.Image(image)}, step=global_step)

  def log_figure(
    self,
    tag: str,
    figure: alt.Chart,
    global_step: int,
  ) -> None:
    bytes_stream = io.BytesIO()
    figure.save(fp=bytes_stream, format="png")
    bytes_stream.seek(0)
    img = Image.open(bytes_stream).convert("RGB")
    self.run.log({tag: wandb.Image(img)}, step=global_step)

  def log_embedding(
    self,
    tag: str,
    mat: np.ndarray,
    metadata: list[str] | None,
    label_img: np.ndarray | None,
    global_step: int,
  ) -> None:
    columns: list[str] = []
    if metadata is not None:
      columns.append("metadata")
    if label_img is not None:
      columns.append("image")
    columns.extend(f"e{i}" for i in range(mat.shape[1]))

    rows: list[list[object]] = []
    for i, vec in enumerate(mat):
      row: list[object] = []
      if metadata is not None:
        row.append(metadata[i])
      if label_img is not None:
        row.append(wandb.Image(label_img[i]))
      row.extend(vec.tolist())
      rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    self.run.log({tag: table}, step=global_step)

  def log_hyperparams(
    self,
    params_dict: dict[str, str | int | float],
  ) -> None:
    self.run.config.update(params_dict)
