import unittest


@unittest.skip(
    "TrainerModule API has changed: train_model and load_from_checkpoint are not yet implemented."
)
class TestBuildTrainer(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
