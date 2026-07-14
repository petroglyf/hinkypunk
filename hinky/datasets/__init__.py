"""Datasets module for JAX Trainer."""
from hinky.datasets.data_struct import DatasetModule
from hinky.datasets.dataset_constructor import (
  build_huggingface_dataset,
  HuggingFaceDatasetConfig,
)

build_dataset_module = build_huggingface_dataset

__all__ = [
  "DatasetModule",
  "HuggingFaceDatasetConfig",
  "build_dataset_module",
  "build_huggingface_dataset",
]