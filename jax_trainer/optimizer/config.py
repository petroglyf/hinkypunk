from pydantic import BaseModel, Field
from enum import Enum


class OptimizerOptions(Enum):
  ADAM = "ADAM"
  ADAMW = "ADAMW"
  SGD = "SGD"


class LRSchedulerEnum(Enum):
  CONSTANT = "constant"
  COSINE_DECAY = "cosine_decay"
  EXPONENTIAL_DECAY = "exponential_decay"
  WARMUP_COSINE_DECAY = "warmup_cosine_decay"


class AdamParams(BaseModel):
  beta1: float = Field(
    0.9, description="Exponential decay rate for the 1st moment estimates."
  )
  beta2: float = Field(
    0.999, description="Exponential decay rate for the 2nd moment estimates."
  )
  eps: float = Field(
    1e-8, description="Term added to the denominator to improve numerical stability."
  )


class AdamwParams(BaseModel):
  beta1: float = Field(
    0.9, description="Exponential decay rate for the 1st moment estimates."
  )
  beta2: float = Field(
    0.999, description="Exponential decay rate for the 2nd moment estimates."
  )
  eps: float = Field(
    1e-8, description="Term added to the denominator to improve numerical stability."
  )
  weight_decay: float = Field(0.01, description="Weight decay coefficient.")


class SGDParams(BaseModel):
  momentum: float = Field(0.0, description="Momentum factor.")
  nesterov: bool = Field(False, description="Whether to use Nesterov momentum.")


class ConstantSchedulerParams(BaseModel):
  value: float = Field(..., description="Constant learning rate value.")


class CosineDecaySchedulerParams(BaseModel):
  alpha: float = Field(
    0.0, description="Minimum learning rate value as a fraction of initial lr."
  )


class WarmupCosineDecaySchedulerParams(BaseModel):
  warmup_steps: int = Field(0, description="Number of warmup steps before decay.", ge=0)
  end_value: float = Field(0.0, description="Final learning rate after decay.")


class ExponentialDecaySchedulerParams(BaseModel):
  cooldown_steps: int = Field(
    0, description="Number of cooldown steps after decay.", ge=0
  )
  warmup_steps: int = Field(0, description="Number of warmup steps before decay.", ge=0)
  decay_rate: float | None = Field(None, description="Decay rate for exponential decay.")
  end_lr: float | None = Field(
    None,
    description="Final learning rate after decay. Either end_lr or end_lr_factor must be set.",
  )
  end_lr_factor: float | None = Field(
    None,
    description="Final learning rate as a factor of initial lr. Either end_lr or end_lr_factor must be set.",
  )
  transition_steps: int = Field(
    1, description="Number of transition steps for exponential decay.", ge=0
  )
  staircase: bool = Field(
    False,
    description="Whether to apply decay in a discrete staircase, as opposed to continuous.",
  )


class SchedulerConfig(BaseModel):
  policy: LRSchedulerEnum = Field(description="Name of the learning rate scheduler.")
  params: (
    ConstantSchedulerParams
    | CosineDecaySchedulerParams
    | ExponentialDecaySchedulerParams
    | WarmupCosineDecaySchedulerParams
  ) = Field(description="Parameters for the learning rate scheduler.")
  decay_steps: int = Field(default=3, description="Number of decay steps.", gt=0)


class RegularlizationConfig(BaseModel):
  grad_clip_norm: float | None = Field(
    None, description="Clip gradients by global norm if set."
  )
  grad_clip_value: float | None = Field(
    None, description="Clip gradients by value if set."
  )
  weight_decay: float = Field(0.0, description="Weight decay coefficient.", gt=0.0)


class OptimizerConfig(BaseModel):
  """Pydantic model for optimizer configuration.

  Attributes:
      name (str): Name of the optimizer.
      lr (float): Learning rate.
      scheduler (dict): Dictionary for learning rate scheduler.
      transforms (dict): Dictionary for gradient transformations.
      params (dict): Dictionary for optimizer parameters.
  """

  name: OptimizerOptions = Field(description="Name of the optimizer.")
  lr: float = Field(default=0.001, description="Learning rate.")

  grad_transforms: RegularlizationConfig = Field(
    description="Dictionary for gradient transformations / regularization."
  )
  params: AdamParams | AdamwParams | SGDParams = Field(
    description="Dictionary for optimizer parameters.",
  )
  scheduler: SchedulerConfig = Field(description="Learning rate scheduler.")
  builder: str = Field("OptimizerBuilder", description="Optimizer builder class.")
