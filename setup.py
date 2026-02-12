import setuptools

setuptools.setup(
  name="jax_trainer",
  version="0.1",
  author="Nick DePalma",
  author_email="ndepalma@googlemail.com",
  description="Lightning-like framework for JAX",
  url="https://github.com/ndepalma/jax-trainer",
  packages=setuptools.find_packages(),
  install_requires=[
    "jax>=0.4.13",
    "jaxlib>=0.4.13",
    "numpy",
    "seaborn",
    "matplotlib",
    "tensorboard>=2.13.0",
    "optax>=0.1.5",
    "orbax-checkpoint>=0.4.0",
    "flax>=0.7.0",
    "absl-py",
    "pydantic>=2.12.5",
    "pydantic-yaml>=1.6.0",
  ],
)
