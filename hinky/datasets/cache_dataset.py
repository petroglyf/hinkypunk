"""CLI tool to build a HuggingFace dataset and cache the processed splits to disk."""

import logging
from pathlib import Path

from cyclopts import App

from hinky.datasets.data_struct import HuggingFaceDatasetConfig, PrepareDatasetConfig
from hinky.datasets.dataset_constructor import process_dataset
from hinky.datasets.dataset_initialization import cache_dataset, get_dataset, split_dataset

_logger = logging.getLogger(__name__)

app = App(help="Build a HuggingFace dataset and cache the processed splits to disk.")


@app.default
def run_cache(
  dataset_config: HuggingFaceDatasetConfig,
  output_dir: Path,
  prepare_config: PrepareDatasetConfig | None = None,
) -> None:
  """Process a HuggingFace dataset and write the resulting splits as cache files.

  Args:
      dataset_config: Dataset construction parameters (batch size, HF URI, etc.).
      prepare_config: Details about how to process the dataset once it's been pulled.
      output_dir: Directory to write the cached dataset splits into.
  """
  if prepare_config is None:
    prepare_config = PrepareDatasetConfig()

  output_dir.mkdir(parents=True, exist_ok=True)

  dataset_module = get_dataset(dataset_config)
  dataset_module_split = split_dataset(prepare_config, dataset_module)
  dataset_module_processed = process_dataset(dataset=dataset_module_split, process_config=prepare_config)
  cache_dataset(dataset_module=dataset_module_processed, output_dir=output_dir)


if __name__ == "__main__":
  app()
