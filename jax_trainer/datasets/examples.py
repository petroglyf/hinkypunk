import numpy as np


from jax_trainer.datasets.data_struct import DatasetModule
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field
from attention_system.datasets.augmentation import normalize_ds
from typing import Annotated, Final
from typing import TypeVar, Generic


class DatasetTransforms(BaseModel):
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  normalize_column: bool = False
  create_validation_set: bool = False
  num_workers: int = 0
  hf_dataset_uri: str


T = TypeVar("T", bound=DatasetTransforms)


def build_huggingface_dataset(dataset_config: T, batch_size: int) -> DatasetModule:
  """Get a dataset from huggingface."""
  ds_train: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="train")  # type: ignore[reportAssignmentType]
  ds_test: Dataset = load_dataset(dataset_config.hf_dataset_uri, split="test")  # type: ignore[reportAssignmentType]
  ds_train.set_format(type="numpy")
  ds_test.set_format(type="numpy")
  ds_train.shuffle()
  ds_test.shuffle()

  ds_validation: Dataset | None = None
  if True:
    ds_validation = load_dataset(dataset_config.hf_dataset_uri, split="validation")  # type: ignore[reportAssignmentType]
    assert ds_validation is not None
    ds_validation.set_format(type="numpy")
    ds_validation.shuffle()

  # Normalize the images
  if True:
    ds_train = normalize_ds(ds_train)
    ds_test = normalize_ds(ds_test)
    ds_validation = (
      normalize_ds(ds_validation) if True and ds_validation is not None else None
    )

  batched_train_iterator = ds_train.batch(batch_size, num_proc=10)
  batched_test_iterator = ds_test.batch(batch_size, num_proc=10)
  batched_validation_iterator = (
    ds_validation.batch(batch_size, num_proc=10) if ds_validation is not None else None
  )

  return DatasetModule(
    dataset_config,
    batched_train_iterator,
    batched_test_iterator,
    batched_validation_iterator,
  )

  # return DatasetStructure(
  #     name=huggingface_uri,
  #     training_set=batched_train_iterator,
  #     validation_set=batched_validation_iterator,
  #     test_set=batched_test_iterator,
  # )


# def build_mnist_datasets(dataset_config: ConfigDict):
#     """Builds MNIST datasets.

#     Args:
#         dataset_config: Configuration for the dataset.

#     Returns:
#         DatasetModule object.
#     """
#     normalize = dataset_config.get("normalize", True)
#     transform = transforms.Compose(
#         [
#             image_to_numpy,
#             normalize_transform(mean=np.array([0.1307]), std=np.array([0.3081]))
#             if normalize
#             else transforms.Lambda(lambda x: x),
#         ]
#     )

#     # Loading the training/validation set
#     train_dataset = MNIST(
#         root=dataset_config.data_dir, train=True, transform=transform, download=True
#     )
#     val_size = dataset_config.get("val_size", 5000)
#     split_seed = dataset_config.get("split_seed", 42)
#     train_set, val_set = data.random_split(
#         train_dataset,
#         [60000 - val_size, val_size],
#         generator=torch.Generator().manual_seed(split_seed),
#     )
#     # Loading the test set
#     test_set = MNIST(
#         root=dataset_config.data_dir, train=False, transform=transform, download=True
#     )

#     train_loader, val_loader, test_loader = build_data_loaders(
#         train_set,
#         val_set,
#         test_set,
#         train=[True, False, False],
#         collate_fn=build_batch_collate(SupervisedBatch),
#         config=dataset_config,
#     )

#     return DatasetModule(
#         dataset_config,
#         train_set,
#         val_set,
#         test_set,
#         train_loader,
#         val_loader,
#         test_loader,
#     )
