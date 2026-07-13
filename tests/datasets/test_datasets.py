import unittest

from jax_trainer.datasets import DatasetModule, build_dataset_module
from jax_trainer.datasets.dataset_constructor import HuggingFaceDatasetConfig


class TestBuildDatasets(unittest.TestCase):
    @unittest.skip("Requires network access and HuggingFace dataset download")
    def test_build_dataset(self):
        config = HuggingFaceDatasetConfig(
            hf_dataset_uri="cifar10",
            batch_size=128,
        )
        dataset_module = build_dataset_module(config)
        self.assertIsInstance(dataset_module, DatasetModule)
        for split in [dataset_module.train, dataset_module.val, dataset_module.test]:
            if split is None:
                continue
            batch = next(iter(split))
            self.assertIn("image", batch)


if __name__ == "__main__":
    unittest.main()
