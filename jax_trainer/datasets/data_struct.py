from typing import Generic, TypeVar

from datasets import Dataset
from flax.struct import dataclass
from pydantic import BaseModel

ConfigType = TypeVar("ConfigType", bound=BaseModel)


@dataclass
class DatasetModule(Generic[ConfigType]):
  """Data module class that holds the datasets and data loaders."""

  config: ConfigType
  train: Dataset
  test: Dataset
  val: Dataset | None = None
  metadata: dict | None = None
