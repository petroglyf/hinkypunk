"""Data structures for datasets."""
from pathlib import Path
from typing import Annotated, Generic, TypeVar

from datasets.table import ConcatenationTable, InMemoryTable, MemoryMappedTable
from pyarrow import Table
from pydantic import BaseModel, ConfigDict, Field

ConfigType = TypeVar("ConfigType", bound=BaseModel)
PermissibleHFTables = ConcatenationTable | InMemoryTable | MemoryMappedTable
PermissibleArrowTables = Table
TableType = TypeVar("TableType", bound=PermissibleHFTables | PermissibleArrowTables)

class DatasetModule(BaseModel, Generic[ConfigType, TableType]):
  """Data module class that holds the datasets and data loaders."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  config: ConfigType
  train: TableType
  test: TableType
  val: TableType | None = None
  metadata: dict | None = None


class PrepareDatasetConfig(BaseModel):
  normalize_column: bool = False
  create_validation_set: bool = False
  desired_square_resolution: int = 200
  pad_and_resize: bool = False

class HuggingFaceDatasetConfig(BaseModel):
  hf_dataset_uri: str

class CachedDatasetConfig(BaseModel):
  cache_path: Path

class TrainingDatasetConfig(BaseModel):
  limit_to: int | None = None
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  class_names: list[str] = Field(default_factory=list)

class FullDatasetSpecification(BaseModel):
  source: HuggingFaceDatasetConfig | CachedDatasetConfig
  preparation: PrepareDatasetConfig
  training_params: TrainingDatasetConfig
