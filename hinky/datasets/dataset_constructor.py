"""Construct a dataset from the dataset specifications."""

import io
import logging
from typing import Annotated, Any, Final, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_dataset
from numba import njit
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from jax_trainer.datasets.data_struct import DatasetModule

_logger = logging.getLogger(__name__)

DESIRED_SQUARE_RESOLUTION: Final[int] = 200

class HuggingFaceDatasetConfig(BaseModel):
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  class_names: list[str] = Field(default_factory=list)
  normalize_column: bool = False
  pad_images_size: int | None = None
  create_validation_set: bool = False
  num_workers: int = 0
  hf_dataset_uri: str
  limit_to: int | None = None


T = TypeVar("T", bound=HuggingFaceDatasetConfig)

@njit
def __normalize_image(all_images: NDArray) -> NDArray:
  mean = all_images.mean() / 255.0
  std = all_images.std() / 255.0
  norm_images = (all_images - mean) / std
  return norm_images

@njit(parallel=True, nogil=True)
def _normalize_ds(image_tensor: NDArray) -> NDArray:
  """Calculate mean and std on train data."""
  normalized_image_np = np.zeros_like(image_tensor, dtype=np.float32)
  for beg in range(0, len(image_tensor), 3):
    end = min(beg + 3, len(image_tensor))
    normalized_image_np[beg:end] = __normalize_image(image_tensor[beg:end])
  return normalized_image_np

def normalize_ds(ds: Dataset) -> Dataset | DatasetDict:
  """Calculate mean and std on train data."""
  # pyrefly: ignore [missing-attribute]
  np_ro_data_in = ds.column("image").combine_chunks().to_numpy_ndarray()
  _ = _normalize_ds(np_ro_data_in)

  return ds


def _pad_and_resize_image(image_in: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
  image = jnp.array(np.array(Image.open(io.BytesIO(image_in["bytes"]))))
  height = image.shape[0]
  width = image.shape[1]

  pad_height = pad_width = max_dim = 0
  if height > width:
    max_dim = height
    pad_height = 0
  else:
    max_dim = width
    pad_width = 0

  perc = 1.0 / (max_dim / DESIRED_SQUARE_RESOLUTION)
  dest_width = int(width*perc)
  dest_height = int(height*perc)
  if height > width:
    pad_width = (DESIRED_SQUARE_RESOLUTION - dest_width) // 2
  else:
    pad_height = (DESIRED_SQUARE_RESOLUTION - dest_height) // 2

  image_out = jax.image.resize(
    image,
    shape=(dest_height, dest_width, 3),
    method="bilinear",
  )

  image_out = jnp.pad(
    image_out,
    ((pad_height, DESIRED_SQUARE_RESOLUTION - dest_height - pad_height), 
      (pad_width, DESIRED_SQUARE_RESOLUTION - dest_width - pad_width), 
      (0, 0)),
    mode="constant",
  )

  return np.array(image_out, dtype=np.uint8), np.array([perc, pad_width, pad_height], dtype=np.float32)

def pad_and_resize_image(ds: Dataset, which_set: str) -> pa.Table:
  """Pad and resize images in the dataset."""
  _logger.info(_ := f"Padding and resizing images in {which_set}...")
  ds_pd = ds.to_pandas()
  # pyrefly: ignore [bad-index, no-matching-overload]
  images_params_proc = ds_pd["image"].map(_pad_and_resize_image)
  images_np = np.stack([i[0] for i in images_params_proc])
  params_np = np.stack([i[1] for i in images_params_proc])

  table_out = ds.data
  p_image_array = pa.FixedShapeTensorArray.from_numpy_ndarray(images_np)
  p_params_array = pa.FixedShapeTensorArray.from_numpy_ndarray(params_np)
  table_out = table_out.drop("image")
  table_out = table_out.append_column("image", p_image_array)
  table_out = table_out.append_column("params", p_params_array)
  return table_out


def build_huggingface_dataset(dataset_config: T) -> DatasetModule:
  """Get a dataset from huggingface."""
  ds_train: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="train")  # type: ignore[reportAssignmentType]
  ds_test: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="validation")  # type: ignore[reportAssignmentType]
  ds_train.set_format(type="jax")
  ds_test.set_format(type="jax")

  ds_train.shuffle()
  ds_test.shuffle()

  ds_validation: Dataset | None = None
  if dataset_config.create_validation_set:
    # pyrefly: ignore [bad-assignment]
    split_test_set: Dataset = ds_test.train_test_split(test_size=0.3, load_from_cache_file=False)

    # pyrefly: ignore [bad-assignment]
    ds_test = split_test_set["test"]
    ds_test.flatten_indices(keep_in_memory=True)

    # pyrefly: ignore [bad-assignment]
    ds_validation = split_test_set["train"]
    # pyrefly: ignore [missing-attribute]
    ds_validation.flatten_indices(num_proc=8, keep_in_memory=True)


  if dataset_config.limit_to is not None:
    ds_train = ds_train.take(dataset_config.limit_to)
    ds_test = ds_test.take(dataset_config.limit_to)
    if ds_validation is not None:
      ds_validation = ds_validation.take(dataset_config.limit_to)


  # Ensure the size of the images are consistent and pad them if necessary. This is required for batching.
  ds_train = pad_and_resize_image(ds_train, "training set")
  ds_test = pad_and_resize_image(ds_test, "test set")
  ds_validation = (
    pad_and_resize_image(ds_validation, "val set")
    if ds_validation is not None
    else None
  )

  # Normalize the images
  if dataset_config.normalize_column:
    # pyrefly: ignore [bad-argument-type, bad-assignment]
    ds_train = normalize_ds(ds_train)
    # pyrefly: ignore [bad-argument-type, bad-assignment]
    ds_test = normalize_ds(ds_test)

    ds_validation = (
      normalize_ds(ds_validation) # pyrefly: ignore [bad-argument-type, bad-assignment]
      if ds_validation is not None
      else None
    )

  return DatasetModule(
    dataset_config,
    # pyrefly: ignore [bad-argument-type]
    ds_train,
    # pyrefly: ignore [bad-argument-type]
    ds_test,
    # pyrefly: ignore [bad-argument-type]
    ds_validation,
  )
