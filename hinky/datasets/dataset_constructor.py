"""Construct a dataset from the dataset specifications."""


import io
import logging
from functools import partial

import numpy as np
import pyarrow as pa
from datasets.table import InMemoryTable
from numba import njit
from numpy.typing import NDArray
from PIL import Image

from hinky.datasets.data_struct import DatasetModule, PermissibleHFTables, PrepareDatasetConfig

_logger = logging.getLogger(__name__)

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

def normalize_ds(ds: PermissibleHFTables) -> PermissibleHFTables:
  """Calculate mean and std on train data."""
  np_ro_data_in = ds["image"].combine_chunks().to_numpy_ndarray()
  _ = _normalize_ds(np_ro_data_in)

  return ds

def _pad_and_resize_image(image_in: dict[str, bytes | str], desired_square_resolution: int) -> tuple[np.ndarray, list[float]]:
  pil_image = Image.open(io.BytesIO(image_in["bytes"])) # pyrefly: ignore [bad-argument-type]
  height = pil_image.height
  width = pil_image.width

  pad_height = pad_width = max_dim = 0
  if height > width:
    max_dim = height
    pad_height = 0
  else:
    max_dim = width
    pad_width = 0

  perc = 1.0 / (max_dim / desired_square_resolution)
  dest_width = int(width*perc)
  dest_height = int(height*perc)
  if height > width:
    pad_width = (desired_square_resolution - dest_width) // 2
  else:
    pad_height = (desired_square_resolution - dest_height) // 2

  image_resized = pil_image.resize(
    size=(dest_width, dest_height),
    resample=Image.Resampling.BILINEAR,
  )

  image_out = Image.new(pil_image.mode, (desired_square_resolution, desired_square_resolution), (0,0,0))
  image_out.paste(im=image_resized, box=(pad_width, pad_height))
  np_im = np.asarray(image_out)

  return np_im, [perc, pad_width, pad_height]

def pad_and_resize_image(ds: PermissibleHFTables, square_resolution:int, which_set: str) -> InMemoryTable:
  """Pad and resize images in the dataset."""
  _logger.info(_ := f"Padding and resizing images in {which_set}...")

  ds_pd = ds.to_pandas()
  pad_and_resize_with = partial(_pad_and_resize_image, desired_square_resolution=square_resolution)
  images_params_proc = ds_pd["image"].map(pad_and_resize_with)
  params_np = np.stack([i[1] for i in images_params_proc], dtype=np.float32)

  table_out = ds.table

  np_image_array = np.stack([i[0] for i in images_params_proc])
  arrow_image_array = pa.FixedShapeTensorArray.from_numpy_ndarray(np_image_array)
  arrow_params_array = pa.FixedSizeListArray.from_arrays(pa.array(params_np.flatten()), params_np.shape[1])
  table_out = table_out.drop("image")
  table_out = table_out.append_column("image", arrow_image_array)
  table_out = table_out.append_column("params", arrow_params_array)
  return InMemoryTable(table=table_out)


def process_dataset(dataset: DatasetModule, process_config: PrepareDatasetConfig) -> DatasetModule:
  """Process the dataset and modify as requested."""
  tbl_train = dataset.train
  tbl_test = dataset.test
  tbl_val = dataset.val

  # Ensure the size of the images are consistent and pad them if necessary. This is required for batching.
  if process_config.pad_and_resize:
    tbl_train = pad_and_resize_image(tbl_train, process_config.desired_square_resolution, "training set")
    tbl_test = pad_and_resize_image(tbl_test, process_config.desired_square_resolution, "test set")
    tbl_val = (
      pad_and_resize_image(tbl_val, process_config.desired_square_resolution, "val set")
      if tbl_val is not None
      else None
    )

  # Normalize the images
  if process_config.normalize_column:
    tbl_train = normalize_ds(tbl_train)
    tbl_test = normalize_ds(tbl_test)

    tbl_val = (
      normalize_ds(tbl_val)
      if tbl_val is not None
      else None
    )

  return DatasetModule(
    config=dataset.config,
    train=tbl_train,
    test=tbl_test,
    val=tbl_val,
  )

# def build_huggingface_dataset(dataset_config: FullDatasetSpecification) -> DatasetModule:
#   """Get a dataset from huggingface."""
#   initial_dataset = get_dataset(
#     source_config=dataset_config.training_params, 
#     input_dir=dataset_config.source.cache_path if isinstance(dataset_config.source, CachedDatasetConfig) else None,
#     )
#   if dataset_config.training_params.limit_to is not None:
#     ds_train = tbl_train.take(dataset_config.training_params.limit_to)
#     ds_test = tbl_test.take(dataset_config.training_params.limit_to)
#     if tbl_val is not None:
#       ds_validation = tbl_val.take(dataset_config.training_params.limit_to)
#   minified_dataset = DatasetModule(
#     train=ds_train.
#   )
#   processed_dataset = process_dataset(initial_dataset, dataset_config.preparation)
#   return processed_dataset