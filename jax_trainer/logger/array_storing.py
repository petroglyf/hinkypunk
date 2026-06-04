"""Storing arrays to be logged for later."""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging


@dataclass
class ArraySpec:
  shape: tuple[int, ...]
  dtype: Any
  device: Any
  value: Any = 0


def array_to_spec(array: jnp.ndarray) -> ArraySpec:
  """Array to spec mutator."""
  return ArraySpec(
      shape=array.shape,
      dtype=array.dtype,
      device=repr(array.device),
      value=array.reshape(-1)[0].item(),
  )


def np_array_to_spec(array: np.ndarray) -> ArraySpec:
  """Numpy array to spec mutator."""
  return ArraySpec(
      shape=array.shape, dtype=array.dtype, device="numpy", value=array.reshape(-1)[0],
  )


def spec_to_array(spec: ArraySpec) -> jnp.ndarray:
  """Mutator back from spec to jax array."""
  device = spec.device
  if device == "numpy":
    return jnp.full(spec.shape, spec.value, dtype=spec.dtype)
  # else
  array = jnp.full(spec.shape, spec.value, dtype=spec.dtype)
  if isinstance(device, str):
    if ":" in device:
      backend_name, device_id = device.split(":")
      device_id = int(device_id)
    else:
      backend_name, device_id = device, 0
    try:
      device = jax.devices(backend_name)[device_id]
    except KeyError:
      logging.warning(f"Backend {backend_name} not found, using CPU instead.")
      device = jax.devices("cpu")[0]
  array = jax.device_put(x=array, device=device)
  return array


def convert_to_array_spec(input_array: jnp.ndarray | np.ndarray) -> ArraySpec:
  """Mutates any tensor into an array spec."""
  if isinstance(input_array, jnp.ndarray):
    return array_to_spec(input_array)
  if isinstance(input_array, np.ndarray):
    return np_array_to_spec(input_array)

  return input_array


def convert_from_array_spec(input_spec: ArraySpec) -> jnp.ndarray:
  """Converts an array spec into a jax tensor."""
  if isinstance(input_spec, ArraySpec):
    return spec_to_array(input_spec)

  return input_spec


def save_pytree(pytree: Any, path: str | Path) -> None:  # noqa: ANN401
  """Pickle dumps a pytree to path."""
  pytree = jax.tree.map(convert_to_array_spec, pytree)
  with Path(path).open("wb") as f:
    pickle.dump(pytree, f)


def load_pytree(path: str | Path) -> Any:  # noqa: ANN401
  """Loads a pytree from a file located at path."""
  with Path(path).open("rb") as f:
    pytree = pickle.load(f)  # noqa: S301
  pytree = jax.tree.map(convert_from_array_spec, pytree)
  return pytree
