from typing import Any

from flax.training import train_state


class TrainState(train_state.TrainState):
  # A simple extension of TrainState to also include mutable variables
  # like batch statistics. If a model has no mutable vars, it is None
  mutable_variables: Any = None
