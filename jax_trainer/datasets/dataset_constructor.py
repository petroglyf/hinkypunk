from typing import Annotated, TypeVar

from datasets import Dataset, load_dataset, DatasetDict
from pydantic import BaseModel, Field

from attention_system.datasets.augmentation import normalize_ds
from jax_trainer.datasets.data_struct import DatasetModule


class HuggingFaceDatasetConfig(BaseModel):
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  class_names: list[str] = Field(default_factory=list)
  normalize_column: bool = False
  create_validation_set: bool = False
  num_workers: int = 0
  hf_dataset_uri: str


T = TypeVar("T", bound=HuggingFaceDatasetConfig)


def build_huggingface_dataset(dataset_config: T) -> DatasetModule:
  """Get a dataset from huggingface."""
  ds_train: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="train")  # type: ignore[reportAssignmentType]
  ds_test: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="test")  # type: ignore[reportAssignmentType]
  ds_train.set_format(type="numpy")
  ds_test.set_format(type="numpy")
  ds_train.shuffle()
  ds_test.shuffle()

  # Normalize the images
  if dataset_config.normalize_column:
    ds_train = normalize_ds(ds_train)
    ds_test = normalize_ds(ds_test)

  ds_validation: Dataset | None = None
  if dataset_config.create_validation_set:
    split_test_set: DatasetDict = ds_test.train_test_split(test_size=0.3)

    ds_test = split_test_set["test"]
    ds_validation = split_test_set["train"]

  batched_train_iterator = ds_train.batch(dataset_config.batch_size, num_proc=10)
  batched_test_iterator = ds_test.batch(dataset_config.batch_size, num_proc=10)
  batched_validation_iterator = (
    ds_validation.batch(dataset_config.batch_size, num_proc=10)
    if ds_validation is not None
    else None
  )

  return DatasetModule(
    dataset_config,
    batched_train_iterator,
    batched_test_iterator,
    batched_validation_iterator,
  )
