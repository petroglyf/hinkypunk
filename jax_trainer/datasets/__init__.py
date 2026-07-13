"""Datasets module for JAX Trainer."""
from jax_trainer.datasets.data_struct import DatasetModule
from jax_trainer.datasets.dataset_constructor import (
  build_huggingface_dataset,
)

build_dataset_module = build_huggingface_dataset

__all__ = [
  "DatasetModule",
  "build_dataset_module",
  "build_huggingface_dataset",
]