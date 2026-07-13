import unittest


def flatten_dict(d: dict, separation_mark: str = ".", prefix: str = "") -> dict:
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}{separation_mark}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, separation_mark=separation_mark, prefix=full_key))
        else:
            result[full_key] = value
    return result


class TestLogger(unittest.TestCase):
    def test_flatten_configdict(self) -> None:
        config = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3,
                },
            },
            "f": {
                "g": 4,
            },
        }
        flattened_config = flatten_dict(config, separation_mark=".")
        self.assertEqual(flattened_config["a"], 1)
        self.assertEqual(flattened_config["b.c"], 2)
        self.assertEqual(flattened_config["b.d.e"], 3)
        self.assertEqual(flattened_config["f.g"], 4)
        self.assertEqual(len(flattened_config), 4)

        flatten_config = flatten_dict(config, separation_mark="/")
        self.assertEqual(flatten_config["a"], 1)
        self.assertEqual(flatten_config["b/c"], 2)
        self.assertEqual(flatten_config["b/d/e"], 3)
        self.assertEqual(flatten_config["f/g"], 4)
        self.assertEqual(len(flatten_config), 4)


if __name__ == "__main__":
    unittest.main()
