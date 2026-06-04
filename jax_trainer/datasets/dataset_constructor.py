"""Construct a dataset from the dataset specifications."""

from typing import Annotated, TypeVar

import jax.numpy as jnp
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field

from jax_trainer.datasets.data_struct import DatasetModule


class HuggingFaceDatasetConfig(BaseModel):
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  class_names: list[str] = Field(default_factory=list)
  normalize_column: bool = False
  create_validation_set: bool = False
  num_workers: int = 0
  hf_dataset_uri: str
  limit_to: int | None = None


T = TypeVar("T", bound=HuggingFaceDatasetConfig)


def _normalize_image(all_images: jnp.Array) -> dict[str, jnp.Array]:
  float_stack = all_images.astype(float)
  mean = float_stack.mean() / 255.0
  std = float_stack.std() / 255.0
  norm_images = (float_stack - mean) / std
  return {"n_image": norm_images}


def normalize_ds(ds: Dataset, which_set: str) -> Dataset | DatasetDict:
  """Calculate mean and std on train data."""
  return ds.map(
    _normalize_image,
    keep_in_memory=True,
    input_columns=("image"),
    desc=f"Normalizing images in {which_set}",
  )


def build_huggingface_dataset(dataset_config: T) -> DatasetModule:
  """Get a dataset from huggingface."""
  ds_train: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="train")  # type: ignore[reportAssignmentType]
  ds_test: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="test")  # type: ignore[reportAssignmentType]
  ds_train.set_format(type="jax")
  ds_test.set_format(type="jax")

  ds_train.shuffle()
  ds_test.shuffle()

  ds_validation: Dataset | DatasetDict | None = None
  if dataset_config.create_validation_set:
    split_test_set: Dataset | DatasetDict = ds_test.train_test_split(test_size=0.3)

    # pyrefly: ignore [bad-assignment]
    ds_test = split_test_set["test"]
    # pyrefly: ignore [bad-assignment]
    ds_validation = split_test_set["train"]

  batched_train_iterator = ds_train.batch(dataset_config.batch_size)
  batched_test_iterator = ds_test.batch(dataset_config.batch_size)
  batched_validation_iterator = (
    # pyrefly: ignore [missing-attribute]
    ds_validation.batch(dataset_config.batch_size) if ds_validation is not None else None
  )

  if dataset_config.limit_to is not None:
    # pyrefly: ignore [missing-attribute]
    batched_train_iterator = batched_train_iterator.take(dataset_config.limit_to)
    # pyrefly: ignore [missing-attribute]
    batched_test_iterator = batched_test_iterator.take(dataset_config.limit_to)
    if batched_validation_iterator is not None:
      batched_validation_iterator = batched_validation_iterator.take(dataset_config.limit_to)

  # Normalize the images
  if dataset_config.normalize_column:
    # pyrefly: ignore [bad-argument-type]
    batched_train_iterator = normalize_ds(batched_train_iterator, "training set")
    # pyrefly: ignore [bad-argument-type]
    batched_test_iterator = normalize_ds(batched_test_iterator, "test set")
    batched_validation_iterator = (
      normalize_ds(batched_validation_iterator, "val set")
      if batched_validation_iterator is not None
      else None
    )

  return DatasetModule(
    dataset_config,
    # pyrefly: ignore [bad-argument-type]
    batched_train_iterator,
    # pyrefly: ignore [bad-argument-type]
    batched_test_iterator,
    # pyrefly: ignore [bad-argument-type]
    batched_validation_iterator,
  )
