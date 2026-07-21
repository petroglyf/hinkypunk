"""This set of functions is meant to just load the dataset from remote or save/load from a local cache."""
import logging
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from hinky.datasets.data_struct import ConfigType, DatasetModule, HuggingFaceDatasetConfig, PrepareDatasetConfig

_logger = logging.getLogger(__name__)

HFDataset = Dataset | DatasetDict | IterableDataset | IterableDatasetDict

def pull_dataset(source_config: ConfigType) -> DatasetModule:
  """Load dataset from source."""
  if isinstance(source_config, HuggingFaceDatasetConfig):
    _logger.info("Loading dataset from huggingface..")
    dataset_config = cast("HuggingFaceDatasetConfig", source_config)  # pyrefly: ignore[redundant-cast]
    ds_train: HFDataset = load_dataset(dataset_config.hf_dataset_uri, split="train")

    try:
      ds_test: HFDataset | None = load_dataset(dataset_config.hf_dataset_uri, split="test")
    except ValueError:
      try:
        ds_test: HFDataset | None = load_dataset(dataset_config.hf_dataset_uri, split="validation")
      except ValueError:
        ds_test = None

    if not isinstance(ds_train, Dataset) or not isinstance(ds_test, Dataset):
      raise TypeError(_ := f"Huggingface should return a Dataset but got type {type(ds_train)} or {type(ds_test)}!")

    return DatasetModule(
      config=source_config,
      train=ds_train.data,
      test=ds_test.data,
    )
  _logger.error(_:= "Unknown dataset source: " + str(type(source_config)))
  raise TypeError

def get_dataset(source_config: ConfigType, input_dir: Path | None = None) -> DatasetModule:
  """Get a dataset or alternatively get a dataset from the cache."""
  if input_dir is None:
    return pull_dataset(source_config)

  _logger.info("Loading dataset from cache..")
  train_tbl = pq.read_table(train_path) if (train_path := input_dir / "train.parquet").exists() else None
  test_tbl = pq.read_table(test_path) if (test_path := input_dir / "test.parquet").exists() else None
  val_tbl = pq.read_table(val_path) if (val_path := input_dir / "val.parquet").exists() else None
  if train_tbl is None or test_tbl is None:
    _logger.error(_ := f"Training or test set not available in cache directory {input_dir}")
    raise FileNotFoundError

  return DatasetModule(
      config=source_config,
      train=train_tbl,
      test=test_tbl,
      val=val_tbl if val_tbl is not None else None,
    )

def cache_dataset(dataset_module: DatasetModule, output_dir: Path) -> None:
  """Store the dataset in a local directory so you don't need to pull it again."""
  for split_name, table in (
    ("train.parquet", dataset_module.train),
    ("test.parquet", dataset_module.test),
    ("val.parquet", dataset_module.val),
  ):
    if table is None:
      continue
    split_dir = output_dir / split_name
    _logger.info("Caching %s split to %s...", split_name, split_dir)
    tbl_out = table.table
    tbl_out.schema.remove_metadata()
    pq.write_table(tbl_out, split_dir)

def split_dataset(dataset_config: PrepareDatasetConfig, dataset: DatasetModule) -> DatasetModule:
  """Split the dataset into train, test, val if configuration so deems it."""
  if dataset_config.create_validation_set and dataset.val is None:
    _logger.info("Creating a validation set from the test set..")
    ds_train = Dataset(dataset.train)
    ds_test = Dataset(dataset.test)

    ds_train.shuffle()
    ds_test.shuffle()

    split_test_set: Dataset | DatasetDict = ds_test.train_test_split(test_size=0.3, load_from_cache_file=False)

    if not isinstance(split_test_set, DatasetDict):
      _logger.error(_:=f"Unexpected type from train/test splitting: {type(split_test_set)}")
      raise TypeError

    ds_train = ds_train.flatten_indices(num_proc=8, keep_in_memory=True)  # pyrefly: ignore[bad-assignment]

    ds_test: Dataset = split_test_set["test"]
    ds_test = ds_test.flatten_indices(num_proc=8, keep_in_memory=True)  # pyrefly: ignore[bad-assignment]

    ds_validation: Dataset = split_test_set["train"]
    ds_validation = ds_validation.flatten_indices(num_proc=8, keep_in_memory=True)  # pyrefly: ignore[bad-assignment]

    return DatasetModule(
      config=dataset.config,
      train=ds_train.data,
      test=ds_test.data,
      val=ds_validation.data,
    )

  # Otherwise val already exists
  return dataset

