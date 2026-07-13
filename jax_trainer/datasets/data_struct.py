"""Data structures for datasets."""
from dataclasses import dataclass
from datasets.table import MemoryMappedTable
from typing import Generic, TypeVar

from datasets import Dataset
from pydantic import BaseModel

ConfigType = TypeVar("ConfigType", bound=BaseModel)


@dataclass
class DatasetModule(Generic[ConfigType]):
  """Data module class that holds the datasets and data loaders."""

  config: ConfigType
  train: MemoryMappedTable
  test: MemoryMappedTable
  val: MemoryMappedTable | None = None
  metadata: dict | None = None
