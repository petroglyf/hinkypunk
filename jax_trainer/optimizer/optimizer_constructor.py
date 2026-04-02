from collections.abc import Callable
from typing import Any

import optax

from jax_trainer.optimizer.config import (
  AdamParams,
  AdamwParams,
  CosineDecaySchedulerParams,
  ExponentialDecaySchedulerParams,
  LRSchedulerEnum,
  OptimizerConfig,
  OptimizerOptions,
  SGDParams,
  WarmupCosineDecaySchedulerParams,
)


class OptimizerBuilder:
  """Class for building optimizers from config.

  Can be overwritten to add custom optimizers and learning rate schedules.
  """

  def __init__(self, optimizer_config: OptimizerConfig) -> None:
    """Initialize OptimizerBuilder."""
    self.optimizer_config = optimizer_config

  def build_optimizer(self, num_epochs: int = 0, num_train_steps_per_epoch: int = 0):
    """Build optimizer from config.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer.

    Returns:
        optax.GradientTransformation: Optimizer.
    """
    # Build elements of optimizer
    opt_class = self.build_optimizer_function()
    lr_schedule = self.build_lr_scheduler(
      num_epochs=num_epochs,
      num_train_steps_per_epoch=num_train_steps_per_epoch,
    )
    pre_grad_trans, post_grad_trans = self.build_gradient_transformations()
    # Put everything together
    optimizer = optax.chain(*pre_grad_trans, opt_class(lr_schedule), *post_grad_trans)
    return optimizer, lr_schedule

  def build_optimizer_function(self):
    """Build optimizer class function from config.

    By default, it supports Adam, AdamW, and SGD. To add custom optimizers, overwrite the
    function build_extra_optimizer_function.

    Returns:
        Callable: Optimizer class function.
    """
    # Build optimizer class
    optimizer_name = self.optimizer_config.name
    optimizer_params = self.optimizer_config.params
    opt_class = None
    if optimizer_name == OptimizerOptions.ADAM and isinstance(optimizer_params, AdamParams):
      opt_class = lambda sched: optax.adam(  # noqa: E731
        learning_rate=sched,
        b1=optimizer_params.beta1,
        b2=optimizer_params.beta2,
        eps=optimizer_params.eps,
      )
    elif optimizer_name == OptimizerOptions.ADAMW and isinstance(optimizer_params, AdamwParams):
      opt_class = lambda sched: optax.adamw(  # noqa: E731
        learning_rate=sched,
        b1=optimizer_params.beta1,
        b2=optimizer_params.beta2,
        eps=optimizer_params.eps,
        weight_decay=optimizer_params.weight_decay,
      )
    elif optimizer_name == OptimizerOptions.SGD and isinstance(optimizer_params, SGDParams):
      opt_class = lambda sched: optax.sgd(  # noqa: E731
        learning_rate=sched,
        momentum=optimizer_params.momentum,
        nesterov=optimizer_params.nesterov,
      )
    else:
      raise ValueError(_ := f"Unknown optimizer {optimizer_name}.")
    return opt_class

  def build_lr_scheduler(self, num_epochs: int = 3, num_train_steps_per_epoch: int = 1):
    """Build learning rate schedule from config.

    By default, it supports constant, cosine decay, exponential decay, and warmup cosine decay.
    To add custom learning rate schedules, overwrite the function build_extra_lr_scheduler.

    Args:
        num_epochs (int, optional): Number of epochs. Defaults to 0.
        num_train_steps_per_epoch (int, optional): Number of training steps per epoch. Defaults to 0.

    Returns:
        Callable: Learning rate schedule function.
    """
    # Build learning rate schedule
    lr = self.optimizer_config.lr
    scheduler_config = self.optimizer_config.scheduler

    scheduler_name = scheduler_config.policy
    scheduler_params = scheduler_config.params
    decay_steps = scheduler_config.decay_steps

    lr_schedule = None
    if scheduler_name == LRSchedulerEnum.CONSTANT:
      lr_schedule = optax.constant_schedule(lr)
    elif scheduler_name == LRSchedulerEnum.COSINE_DECAY and isinstance(
      scheduler_params, CosineDecaySchedulerParams
    ):
      lr_schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=decay_steps,
        alpha=scheduler_params.alpha,
      )
    elif scheduler_name == LRSchedulerEnum.EXPONENTIAL_DECAY and isinstance(
      scheduler_params, ExponentialDecaySchedulerParams
    ):
      # Exponential decay with cooldown and warmup
      cooldown = scheduler_params.cooldown_steps
      warmup = scheduler_params.warmup_steps
      num_schedule_steps = decay_steps - cooldown - warmup
      if scheduler_params.decay_rate is not None:
        decay_rate = scheduler_params.decay_rate
      else:
        if scheduler_params.end_lr is not None:
          end_lr_factor = scheduler_params.end_lr / lr
        elif scheduler_params.end_lr_factor is not None:
          end_lr_factor = scheduler_params.end_lr_factor
        else:
          raise ValueError(_ := "Either end_lr or end_lr_factor must be specified.")
        decay_rate = end_lr_factor ** (1.0 / num_schedule_steps)
      lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        decay_rate=decay_rate,
        warmup_steps=warmup,
        transition_steps=scheduler_params.transition_steps,
        staircase=scheduler_params.staircase,
      )
      if cooldown > 0:
        end_lr = lr * (decay_rate**num_schedule_steps)
        lr_schedule = optax.join_schedules(
          schedules=[
            lr_schedule,
            optax.linear_schedule(
              init_value=end_lr,
              end_value=0.0,
              transition_steps=cooldown,
            ),
          ],
          boundaries=[decay_steps - cooldown],
        )
    elif scheduler_name == LRSchedulerEnum.WARMUP_COSINE_DECAY and isinstance(
      scheduler_params, WarmupCosineDecaySchedulerParams
    ):
      lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        decay_steps=decay_steps,
        warmup_steps=scheduler_params.warmup_steps,
        end_value=scheduler_params.end_value,
      )

    return lr_schedule

  def build_gradient_transformations(self):
    """Build gradient transformations from config.

    By default, it supports gradient clipping by norm and value, and weight decay. To add custom
    gradient transformations, overwrite this function and call super().build_gradient_transformations()
    in it. We distinguish between pre- and post-optimizer gradient transformations. Pre-optimizer
    gradient transformations are applied before the optimizer, e.g. gradient clipping. Post-optimizer
    gradient transformations are applied after the optimizer.

    Returns:
        Tuple[List[Callable], List[Callable]]: Tuple of pre-optimizer and post-optimizer gradient transformations.
    """
    # Gradient transformation
    optimizer_name = self.optimizer_config.name
    transform_config = self.optimizer_config.grad_transforms
    grad_trans = {"pre": [], "post": []}

    def add_grad_trans(config: Any, gt_fn: Callable):
      # if isinstance(config, (float, int, str, bool)):
      gt = gt_fn(config)  # clip
      grad_trans["pre"].append(gt)
      # else:
      #   gt = gt_fn(config.value)  # clip_by_global_norm
      #   if config.get("before_optimizer", True):
      #     grad_trans["pre"].append(gt)
      #   else:
      #     grad_trans["post"].append(gt)

    if transform_config.grad_clip_norm is not None:
      add_grad_trans(transform_config.grad_clip_norm, optax.clip_by_global_norm)
    if transform_config.grad_clip_value is not None:
      add_grad_trans(transform_config.grad_clip_value, optax.clip)
    if transform_config.weight_decay > 0.0 and optimizer_name != OptimizerOptions.ADAMW:
      add_grad_trans(transform_config.weight_decay, optax.add_decayed_weights)

    return grad_trans["pre"], grad_trans["post"]
