import unittest

import optax

from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.optimizer.config import (
    AdamParams,
    AdamwParams,
    ConstantSchedulerParams,
    CosineDecaySchedulerParams,
    ExponentialDecaySchedulerParams,
    LRSchedulerEnum,
    OptimizerConfig,
    OptimizerOptions,
    RegularlizationConfig,
    SchedulerConfig,
    SGDParams,
    WarmupCosineDecaySchedulerParams,
)

NO_GRAD_TRANSFORMS = RegularlizationConfig()
CONSTANT_SCHEDULER = SchedulerConfig(
    policy=LRSchedulerEnum.CONSTANT,
    params=ConstantSchedulerParams(),
)


def build_optimizer(optimizer_config: OptimizerConfig):
    optimizer_builder = OptimizerBuilder(optimizer_config)
    return optimizer_builder.build_optimizer(num_epochs=100, num_train_steps_per_epoch=1000)


class TestBuildOptimizer(unittest.TestCase):
    def test_build_optimizer_sgd(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.SGD,
            lr=0.001,
            params=SGDParams(momentum=0.9, nesterov=True),
            scheduler=CONSTANT_SCHEDULER,
            grad_transforms=NO_GRAD_TRANSFORMS,
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_adam(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAM,
            lr=0.001,
            params=AdamParams(beta1=0.9),
            scheduler=CONSTANT_SCHEDULER,
            grad_transforms=NO_GRAD_TRANSFORMS,
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_adamw(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAMW,
            lr=0.001,
            params=AdamwParams(beta1=0.9),
            scheduler=CONSTANT_SCHEDULER,
            grad_transforms=NO_GRAD_TRANSFORMS,
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_schedule_cosine_decay(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAM,
            lr=0.001,
            params=AdamParams(beta1=0.9),
            scheduler=SchedulerConfig(
                policy=LRSchedulerEnum.COSINE_DECAY,
                params=CosineDecaySchedulerParams(alpha=0.1),
                decay_steps=1000,
            ),
            grad_transforms=NO_GRAD_TRANSFORMS,
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_schedule_exponential_decay(self) -> None:
        for end_kwarg in [
            {"decay_rate": 0.1},
            {"end_lr": 0.0001},
            {"end_lr_factor": 0.1},
        ]:
            for warmup_steps in [0, 100]:
                for cooldown_steps in [0, 10]:
                    params = ExponentialDecaySchedulerParams(
                        transition_steps=1,
                        staircase=False,
                        warmup_steps=warmup_steps,
                        cooldown_steps=cooldown_steps,
                        **end_kwarg,
                    )
                    config = OptimizerConfig(
                        name=OptimizerOptions.ADAM,
                        lr=0.001,
                        params=AdamParams(beta1=0.9),
                        scheduler=SchedulerConfig(
                            policy=LRSchedulerEnum.EXPONENTIAL_DECAY,
                            params=params,
                            decay_steps=1000,
                        ),
                        grad_transforms=NO_GRAD_TRANSFORMS,
                    )
                    optimizer, _ = build_optimizer(config)
                    self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_schedule_warmup_cosine_decay(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAM,
            lr=0.001,
            params=AdamParams(beta1=0.9),
            scheduler=SchedulerConfig(
                policy=LRSchedulerEnum.WARMUP_COSINE_DECAY,
                params=WarmupCosineDecaySchedulerParams(warmup_steps=100),
                decay_steps=1000,
            ),
            grad_transforms=NO_GRAD_TRANSFORMS,
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_gradient_clipping(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAM,
            lr=0.001,
            params=AdamParams(beta1=0.9),
            scheduler=CONSTANT_SCHEDULER,
            grad_transforms=RegularlizationConfig(grad_clip_norm=1.0, grad_clip_value=0.1),
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

    def test_build_optimizer_weight_decay(self) -> None:
        config = OptimizerConfig(
            name=OptimizerOptions.ADAM,
            lr=0.001,
            params=AdamParams(beta1=0.9),
            scheduler=CONSTANT_SCHEDULER,
            grad_transforms=RegularlizationConfig(weight_decay=0.01),
        )
        optimizer, _ = build_optimizer(config)
        self.assertIsInstance(optimizer, optax.GradientTransformation)


if __name__ == "__main__":
    unittest.main()
