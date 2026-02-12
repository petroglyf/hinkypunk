# Standard libraries
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import jax

# JAX/Flax libraries
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from dltype import FloatTensor, dltyped
from flax import nnx
from flax.core import FrozenDict
from pydantic import create_model

# ML collections for config
from tabulate import tabulate as python_tabulate
from tqdm.auto import tqdm

from jax_trainer.datasets import DatasetModule
from jax_trainer.datasets.dataset_constructor import HuggingFaceDatasetConfig
from jax_trainer.logger import (
  HostMetrics,
  ImmutableMetrics,
  LogFreq,
  LogMetricMode,
  update_metrics,
)
from jax_trainer.logger.config import LoggerConfig
from jax_trainer.optimizer.config import OptimizerConfig
from jax_trainer.optimizer.optimizer_constructor import OptimizerBuilder
from jax_trainer.trainer.checkpoint_kit import ModelCheckpoint
from jax_trainer.trainer.config import ModelConfig, TrainerConfig
from jax_trainer.trainer.state import TrainState
from jax_trainer.utils import flatten_dict, resolve_import

ModelParamsType = TypeVar("ModelParamsType")


class TrainerModule(Generic[ModelParamsType]):
  def __init__(
    self,
    trainer_config: TrainerConfig,
    model_config: ModelConfig[ModelParamsType],
    optimizer_config: OptimizerConfig,
    dataset_config: HuggingFaceDatasetConfig,
    dataset: DatasetModule,
  ):
    """A basic Trainer module summarizing most common training functionalities like logging, model initialization, training loop, etc.

    Args:
        trainer_config: A dictionary containing the trainer configuration.
        model_config: A dictionary containing the model configuration.
        optimizer_config: A dictionary containing the optimizer configuration.
        exmp_input: An input to the model with which the shapes are inferred.
    """
    super().__init__()
    self.trainer_config = trainer_config
    self.model_config = model_config
    self.optimizer_config = optimizer_config
    self.dataset = dataset
    self.dataset_config = dataset_config

    # Default properties for trainer config
    self.trainer_config.check_val_every_n_epoch = (
      self.trainer_config.check_val_every_n_epoch
    )
    self.prepare_rngs()

    # Create empty model. Note: no parameters yet

    self.build_model(model_config)
    # Init trainer parts
    self.create_jitted_functions()
    # self.init_model()
    self.init_logger(self.trainer_config.logger)
    self.init_callbacks()
    self.checkpoint_manager = None
    if trainer_config.checkpoint_config is not None:
      self.checkpoint_manager = ModelCheckpoint(
        params_config=trainer_config.checkpoint_config, trainer=self
      )

  def batch_to_input(self, batch: dict[str, jax.Array]) -> dict[str, Any]:
    raise NotImplementedError

  def prepare_rngs(self):
    root_rng_train = jax.random.key(self.trainer_config.seed)
    root_rng_eval = jax.random.key(self.trainer_config.seed_eval)
    n_rngs = len(self.trainer_config.rngs)

    # Initialize the rngs for the training pass
    rngs_seeds_train = jax.random.split(root_rng_train, n_rngs + 1)
    rngs_params_train = dict(
      zip(["params", *self.trainer_config.rngs], rngs_seeds_train, strict=False),
    )
    self.rngs_train = nnx.Rngs(**rngs_params_train)

    # Initialize the rngs for the evaluation pass
    rngs_seeds_eval = jax.random.split(root_rng_eval, n_rngs + 1)
    rngs_params_eval = dict(
      zip(["params", *self.trainer_config.rngs], rngs_seeds_eval, strict=False),
    )
    self.rngs_eval = nnx.Rngs(**rngs_params_eval)

  def build_model(self, model_config: ModelConfig[ModelParamsType]):
    """Creates the model class from the model_config.

    Args:
        model_config: A dictionary containing the model configuration.
    """
    # Create model
    model_class = resolve_import(model_config.name)
    hparams = model_config.hparams
    self.model = model_class(**hparams, rngs=self.rngs_train)

  def init_logger(self, logger_config: LoggerConfig):
    """Initializes a logger and creates a logging directory.

    Args:
        logger_params: A dictionary containing the specification of the logger.
    """
    all_config_type = create_model(
      "FullModelConfig",
      trainer=TrainerConfig,
      model=ModelConfig[ModelParamsType],
      optimizer=OptimizerConfig,
    )
    all_config = all_config_type(
      trainer=self.trainer_config,
      model=self.model_config,
      optimizer=self.optimizer_config,
    )
    LoggerClass = resolve_import(logger_config.class_name)
    self.logger = LoggerClass(logger_config, self.model_config.name)
    # Log the hyperparams
    # hparams = flatten_configdict(all_config)
    # hparams = jax.tree_map(class_to_name, hparams)
    # logger.log_hyperparams(hparams)
    return

    # Save config and exmp_input
    log_dir = self.logger.log_dir
    logging.info(f"Logging at {log_dir}")
    self.log_dir = log_dir
    self.trainer_config.logger.log_dir = log_dir
    os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(
      log_dir=log_dir, program_name="absl_logging"
    )
    logging.set_verbosity(logger_config.log_file_verbosity)
    logging.set_stderrthreshold(logger_config.stderrthreshold)
    if not os.path.isfile(os.path.join(log_dir, "config.yaml")):
      yaml_str = to_yaml_str(all_config)
      with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        f.write(yaml_str)
    # if not os.path.isfile(os.path.join(log_dir, "exmp_input.pkl")):
    #   save_pytree(self.exmp_input, os.path.join(log_dir, "exmp_input.pkl"))
    if self.trainer_config.tabulate_model:
      tab = self.tabulate(self.exmp_input)
      logging.info("Model summary:\n" + tab)
      with open(os.path.join(log_dir, "model.txt"), "w") as f:
        f.write(tab)
    if self.trainer_config.tabulate_params:
      tab = self.tabulate_params()
      logging.info("Parameter summary:\n" + tab)
      with open(os.path.join(log_dir, "params.txt"), "w") as f:
        f.write(tab)

  def init_callbacks(self):
    """Initializes the callbacks defined in the trainer config."""
    self.callbacks = []
    self.train_step_callbacks = []
    callback_configs = self.trainer_config.callbacks

    for callback_config in callback_configs:
      logging.info(f"Initializing callback {callback_config.name}")
      # if callback_config.get("class_name", None) is not None:
      callback_class = resolve_import(callback_config.class_name)
      callback_class_config = callback_class.encapsulate_config(callback_config.options)
      # else:
      #   callback_class = getattr(callbacks, name)
      callback = callback_class(
        params_config=callback_class_config, callback_config=callback_config, trainer=self
      )
      self.callbacks.append(callback)
      if hasattr(callback, "on_training_step"):
        self.train_step_callbacks.append(callback)

  def init_model(self):
    """Creates an initial training state with newly generated network parameters.

    Args:
        exmp_input: An input to the model with which the shapes are inferred.
    """
    # Run model initialization
    # variables = self.run_model_init()
    # if isinstance(variables, FrozenDict):
    #   mutable_variables, params = variables.pop("params")
    # else:
    #   params = variables.pop("params")
    #   mutable_variables = variables
    # if len(mutable_variables) == 0:
    #   mutable_variables = None
    # Create default state. Optimizer is initialized later
    # self.state = TrainState(
    #   step=0,
    #   apply_fn=self.model.__call__,
    #   params=nnx.state(self.model, nnx.Param),
    #   # mutable_variables=mutable_variables,
    #   tx=None,
    #   opt_state=None,
    # )
    self.state = nnx.Optimizer(self.model, optax_optim)

  def init_train_metrics(self, batch: dict[str, jax.Array] | None = None) -> FrozenDict:
    if not hasattr(self, "train_metric_shapes"):
      self.train_metric_shapes = None
    if self.train_metric_shapes is None:
      # if batch is None:
      #   batch = self.exmp_input
      _, self.train_metric_shapes = nnx.eval_shape(
        self.train_step,
        optimizer_state=self.state,
        model_kwargs=batch,
        metrics=None,
        rngs=self.rngs_train,
      )
    return jax.tree_map(lambda x: jnp.zeros_like(x), self.train_metric_shapes)

  def init_eval_metrics(self, batch: dict[str, Any] | None = None) -> FrozenDict:
    if not hasattr(self, "eval_metric_shapes"):
      self.eval_metric_shapes = None
    if self.eval_metric_shapes is None:
      if batch is None:
        batch = self.exmp_input
      self.eval_metric_shapes = nnx.eval_shape(
        self.eval_step,
        optimizer_state=self.state,
        model_kwargs=batch,
        metrics=None,
        rngs=self.rngs_eval,
      )
    return jax.tree_map(lambda x: jnp.zeros_like(x), self.eval_metric_shapes)

  def set_dataset(self, dataset: DatasetModule):
    for callback in self.callbacks:
      callback.set_dataset(dataset)
    self.dataset = dataset

  # def run_model_init(self, exmp_input: Batch, init_rng: jax.Array) -> dict:
  #   """The model initialization call.

  #   Args:
  #       exmp_input: An input to the model with which the shapes are inferred.
  #       init_rng: A jax.random.PRNGKey.

  #   Returns:
  #       The initialized variable dictionary.
  #   """
  #   exmp_input = self.batch_to_input(exmp_input)
  #   variables = self.model.init(rngs, exmp_input, train=True)
  #   if not isinstance(variables, FrozenDict):
  #     variables = freeze(variables)
  #   return variables

  def tabulate(self, exmp_input: dict[str, jax.Array]) -> str:
    """Prints a summary of the Module represented as table.

    Args:
        exmp_input: An input to the model with which the shapes are inferred.
    """
    rngs = self.get_model_rng(random.PRNGKey(0))
    exmp_input = self.batch_to_input(exmp_input)
    return self.model.tabulate(
      rngs, exmp_input, train=True, console_kwargs={"force_terminal": False, "width": 300}
    )

  def tabulate_params(self) -> str:
    """Prints a summary of the parameters represented as table.

    Args:
        exmp_input: An input to the model with which the shapes are inferred.
    """
    params = self.state.params
    params = flatten_dict(params)
    param_shape = jax.tree_map(lambda x: x.shape, params)
    param_count = jax.tree_map(lambda x: np.prod(x.shape), params)
    param_dtype = jax.tree_map(lambda x: x.dtype, params)
    param_mean = jax.tree_map(lambda x: jnp.mean(x).item(), params)
    param_std = jax.tree_map(lambda x: jnp.std(x).item(), params)
    param_min = jax.tree_map(lambda x: jnp.min(x).item() if x.size > 0 else 0, params)
    param_max = jax.tree_map(lambda x: jnp.max(x).item() if x.size > 0 else 0, params)
    summary = defaultdict(list)
    for key in sorted(list(params.keys())):
      summary["Name"].append(key)
      summary["Shape"].append(param_shape[key])
      summary["Count"].append(param_count[key])
      summary["Dtype"].append(param_dtype[key])
      summary["Mean"].append(param_mean[key])
      summary["Std"].append(param_std[key])
      summary["Min"].append(param_min[key])
      summary["Max"].append(param_max[key])
    return python_tabulate(summary, headers="keys")

  def init_optimizer(self, num_epochs: int, num_train_steps_per_epoch: int):
    """Initializes the optimizer and learning rate scheduler.

    Args:
        num_epochs: Number of epochs the model will be trained for.
        num_train_steps_per_epoch: Number of training steps per epoch.
    """
    builder = OptimizerBuilder(self.optimizer_config)
    optimizer, lr_schedule = builder.build_optimizer(
      num_epochs=num_epochs,
      num_train_steps_per_epoch=num_train_steps_per_epoch,
    )
    self.lr_schedule = lr_schedule  # Save for logging
    # Initialize training state
    # self.state = TrainState.create(
    #   apply_fn=self.state.apply_fn,
    #   params=self.state.params,
    #   mutable_variables=self.state.mutable_variables,
    #   tx=optimizer,
    #   # rng=self.state.rng,
    # )
    self.state = nnx.Optimizer(self.model, optimizer)
    # self.state = self.state.replace(step=jnp.array(self.state.step))  # Convert to jnp.array for compiling.
    # self.state = jax.device_put(self.state)

  def create_jitted_functions(self):
    """Creates jitted versions of the training and evaluation functions.

    If self.debug is True, not jitting is applied.
    """
    train_step, eval_step = self.create_functions()
    if self.trainer_config.debug:  # Skip jitting
      print("Skipping jitting due to debug=True")
      self.train_step = train_step
      self.eval_step = eval_step
    else:  # Jit
      print("Not skipping jitting due to debug=True")
      train_donate_argnames = ["metrics"]  # Donate metrics to avoid copying.
      if self.trainer_config.donate_train_state:
        train_donate_argnames.append("optimizer_state")
      self.train_step = nnx.jit(
        train_step,
        # donate_argnames=["metrics"],
      )
      self.eval_step = nnx.jit(
        eval_step,
        # donate_argnames=["metrics"],  # Donate metrics to avoid copying.
      )

  def loss_function(
    self,
    model: nnx.Module,
    batch: dict[str, jax.Array],
    rngs: nnx.Rngs,
    train: bool = True,
  ) -> tuple[jax.Array, ImmutableMetrics]:
    """The loss function that is used for training.

    This function needs to be overwritten by a subclass.
    """
    raise NotImplementedError

  # def model_apply(
  #   self,
  #   params: Any,
  #   state: TrainState,
  #   input: Any,
  #   rng: jax.Array,
  #   train: bool = True,
  #   mutable: list[str] | None = None,
  #   **kwargs,
  # ) -> tuple[Any, dict | None]:
  #   """The model apply function that can be used in the loss function for simplification."""
  #   rngs = self.get_model_rng(rng)
  #   variables = {"params": params}
  #   mutable_keys = [] if mutable is None else mutable
  #   if state.mutable_variables is not None:
  #     variables.update(
  #       {k: state.mutable_variables[k] for k in state.mutable_variables.keys()}
  #     )
  #     if train:
  #       mutable_keys += list(state.mutable_variables.keys())
  #   if len(mutable_keys) == 0:
  #     mutable_keys = False
  #   out = state.apply_fn(
  #     variables, input, train=train, rngs=rngs, mutable=mutable_keys, **kwargs
  #   )
  #   if mutable_keys is not False:
  #     out, mutable_vars = out
  #   else:
  #     mutable_vars = None
  #   return out, mutable_vars

  def create_training_function(
    self,
  ) -> Callable[
    [nnx.Optimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
    tuple[nnx.Optimizer, ImmutableMetrics],
  ]:
    """Creates and returns a function for the training step.

    The function takes as input the training state and a batch from the train loader. The
    function is expected to return a dictionary of logging metrics, and a new train state.
    """

    def train_step(
      optimizer_state: nnx.Optimizer,
      model_kwargs: dict[str, jax.Array],
      metrics: ImmutableMetrics | None,
      rngs: nnx.Rngs,
    ) -> tuple[nnx.Optimizer, ImmutableMetrics]:
      all_value_and_grad_out = nnx.value_and_grad(self.loss_function, has_aux=True)(
        model=optimizer_state.model,
        batch=model_kwargs,
        rngs=rngs,
        train=True,
      )
      (_, step_metrics), grads = all_value_and_grad_out
      # In place updates. -- this usualy takes a model parameter but the old version is just grads.
      optimizer_state.update(grads)

      if self.trainer_config.log_grad_norm:
        grad_norm = optax.global_norm(grads)
        step_metrics["optimizer/grad_global_norm"] = {
          "value": grad_norm,
          "log_freq": LogFreq.STEP,
        }
        step_metrics["optimizer/grad_global_norm_max"] = {
          "value": grad_norm,
          "mode": LogMetricMode.MAX,
          "log_freq": LogFreq.EPOCH,
        }
        # params_norm = optax.global_norm(state.params)
        # step_metrics["optimizer/params_global_norm"] = {
        #   "value": params_norm,
        #   "log_freq": LogFreq.STEP,
        # }
      metrics = update_metrics(
        metrics,
        step_metrics,
        train=True,
        batch_size=self.dataset_config.batch_size,
      )
      return optimizer_state, metrics

    return train_step

  def create_evaluation_function(
    self,
  ) -> Callable[
    [nnx.Optimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
    ImmutableMetrics,
  ]:
    """Creates and returns a function for the evaluation step.

    The function takes as input the training state and a batch from the val/test loader. The
    function is expected to return a dictionary of logging metrics, and a new train state.
    """

    def eval_step(
      optimizer_state: nnx.Optimizer,
      model_kwargs: dict[str, jax.Array],
      metrics: ImmutableMetrics | None,
      rngs: nnx.Rngs,
    ) -> ImmutableMetrics:
      all_value_and_grad_out = nnx.value_and_grad(self.loss_function, has_aux=True)(
        optimizer_state.model,
        model_kwargs,
        rngs=rngs,
        train=False,
      )
      (_, step_metrics), _ = all_value_and_grad_out
      metrics = update_metrics(
        metrics,
        step_metrics,
        train=False,
        batch_size=self.dataset_config.batch_size,
      )

      return metrics

    return eval_step

  def create_functions(
    self,
  ) -> tuple[
    Callable[
      [nnx.Optimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
      tuple[nnx.Optimizer, ImmutableMetrics],
    ],
    Callable[
      [nnx.Optimizer, dict[str, jax.Array], ImmutableMetrics | None, nnx.Rngs],
      ImmutableMetrics,
    ],
  ]:
    """Creates and returns functions for the training and evaluation step.

    The functions take as input the training state and a batch from the train/ val/test loader.
    Both functions are expected to return a dictionary of logging metrics, and the training
    function a new train state. This function needs to be overwritten by a subclass. The
    train_step and eval_step functions here are examples for the signature of the functions.
    """
    return self.create_training_function(), self.create_evaluation_function()

  def train_model(
    self,
    # train_loader: Iterator,
    # val_loader: Iterator,
    # test_loader: Iterator | None = None,
    datasets: DatasetModule[Any],
    num_epochs: int = 500,
  ) -> dict[str, Any]:
    """Starts a training loop for the given number of epochs.

    Args:
        train_loader: Data loader of the training set.
        val_loader: Data loader of the validation set.
        test_loader: If given, best model will be evaluated on the test set.
        num_epochs: Number of epochs for which to train the model.

    Returns:
        A dictionary of the train, validation and evt. test metrics for the
        best model on the validation set.
    """
    # Create optimizer and the scheduler for the given number of epochs
    self.init_optimizer(num_epochs, len(datasets.train))
    self.global_step = 0
    # Prepare training loop
    self.on_training_start()
    self.test_eval_function(datasets.val)
    all_eval_metrics = {}
    train_metrics = None
    training_failed = False
    for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
      self.on_training_epoch_start(epoch_idx)
      train_metrics, epoch_metrics = self.train_epoch(
        datasets.train, epoch_idx=epoch_idx, train_metrics=train_metrics
      )
      if self.trainer_config.detect_nans:
        nan_keys = self.trainer_config.nan_keys
        if isinstance(nan_keys, str):
          nan_keys = (nan_keys,)
        if any([np.isnan(epoch_metrics.get(key, 0.0)) for key in nan_keys]):
          logging.error(
            f"NaN detected in epoch metrics of epoch {epoch_idx}. Aborting training."
          )
          training_failed = True
          break
      self.on_training_epoch_end(epoch_metrics, epoch_idx)
      # Validation every N epochs
      if (
        self.trainer_config.check_val_every_n_epoch > 0
        and epoch_idx % self.trainer_config.check_val_every_n_epoch == 0
      ):
        self.on_validation_epoch_start(epoch_idx)
        eval_metrics = self.eval_model(datasets.val, mode="val", epoch_idx=epoch_idx)
        all_eval_metrics[epoch_idx] = eval_metrics
        self.on_validation_epoch_end(eval_metrics, epoch_idx)
    if not training_failed:
      self.on_training_end()
      # Test best model if possible
      if datasets.test is not None:
        # self.load_model(raise_if_not_found=False)
        self.on_test_epoch_start(epoch_idx)
        test_metrics = self.eval_model(datasets.test, mode="test", epoch_idx=epoch_idx)
        self.on_test_epoch_end(test_metrics, epoch_idx)
        all_eval_metrics["test"] = test_metrics
    # Close logger
    self.logger.finalize("success" if not training_failed else "failed")
    for callback in self.callbacks:
      callback.finalize("success" if not training_failed else "failed")
    if self.checkpoint_manager is not None:
      self.checkpoint_manager.finalize("success" if not training_failed else "failed")
    return all_eval_metrics

  def test_model(
    self, test_loader: Iterator, apply_callbacks: bool = False, epoch_idx: int = 0
  ) -> dict[str, Any]:
    """Tests the model on the given test set.

    Args:
        test_loader: Data loader of the test set.
        apply_callbacks: If True, the callbacks will be applied.
        epoch_idx: The epoch index to use for the callbacks and logging.
    """
    test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
    if apply_callbacks:
      self.on_test_epoch_end(test_metrics, epoch_idx=epoch_idx)
    return test_metrics

  def test_eval_function(self, val_loader: Iterator) -> None:
    """Tests the evaluation function on a single batch.

    This is useful to check if the functions have the correct signature and return the correct
    values. This prevents annoying errors that occur at the first evaluation step.

    This function does not test the training function anymore. This is because the training
    function is already executed in the first epoch and we change its jit signature to donate
    the train state and metrics. Thus, executing a training step requires updating the train
    state, which we would not want to do here. The compilation time is logged during the very
    first training step.

    Args:
        val_loader: Data loader of the validation set.
    """
    print("Verifying evaluation function...")
    val_batch = self.batch_to_input(next(iter(val_loader)))
    eval_metrics = self.init_eval_metrics(val_batch)
    start_time = time.time()
    logging.info("Testing and compiling eval_step...")
    _ = self.eval_step(self.state, val_batch, eval_metrics, rngs=self.rngs_eval)
    logging.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

  def train_epoch(
    self, train_loader: Iterator, epoch_idx: int, train_metrics: ImmutableMetrics | None
  ) -> tuple[ImmutableMetrics, HostMetrics]:
    """Trains a model for one epoch.

    Args:
        train_loader: Data loader of the training set.
        epoch_idx: Current epoch index.

    Returns:
        A dictionary of the average training metrics over all batches
        for logging.
    """
    # Train model for one epoch, and log avg loss and accuracy
    self.logger.start_epoch(epoch_idx, mode="train")

    for batch in self.tracker(train_loader, desc="Training", leave=False):
      if train_metrics is None:
        train_metrics = self.init_train_metrics(self.batch_to_input(batch))
      if self.global_step == 0:
        # Log compilation and execution time of the first batch.
        logging.info("Compiling train_step...")
        start_time = time.time()
        self.state, train_metrics = self.train_step(
          self.state, self.batch_to_input(batch), train_metrics, rngs=self.rngs_train
        )
        logging.info(
          f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds."
        )
      else:
        # Annotated with step number for TensorBoard profiling.
        with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
          self.state, train_metrics = self.train_step(
            self.state, self.batch_to_input(batch), train_metrics, rngs=self.rngs_train
          )
      for callback in self.train_step_callbacks:
        callback.on_training_step(train_metrics, epoch_idx, self.global_step)
      train_metrics = self.logger.log_step(train_metrics)
      self.global_step += 1
    train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics)
    return train_metrics, epoch_metrics

  def eval_model(self, data_loader: Iterator, mode: str, epoch_idx: int) -> HostMetrics:
    """Evaluates the model on a dataset.

    Args:
        data_loader: Data loader of the dataset to evaluate on.
        mode: Whether 'val' or 'test'
        epoch_idx: Current epoch index.

    Returns:
        A dictionary of the evaluation metrics, averaged over data points
        in the dataset.
    """
    # Test model on all images of a data loader and return avg loss
    self.logger.start_epoch(epoch_idx, mode=mode)
    eval_metrics = self.init_eval_metrics()
    step_count = 0
    for batch in self.tracker(data_loader, desc=mode.capitalize(), leave=False):
      eval_metrics = self.eval_step(
        self.state, self.batch_to_input(batch), eval_metrics, rngs=self.rngs_eval
      )
      step_count += 1
    if step_count == 0:
      logging.warning(f"No batches in {mode} loader at epoch {epoch_idx}.")
    _, metrics = self.logger.end_epoch(eval_metrics, save_metrics=True)
    return metrics

  def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
    """Wraps an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

    Args:
        iterator: Iterator to wrap in tqdm.
        kwargs: Additional arguments to tqdm.

    Returns:
        Wrapped iterator if progress bar is enabled, otherwise same iterator
        as input.
    """
    if self.trainer_config.enable_progress_bar:
      return tqdm(iterator, **kwargs)
    else:
      return iterator

  def on_training_start(self):
    """Method called before training is started.

    Can be used for additional initialization operations etc.
    """
    logging.info("Starting training")
    for callback in self.callbacks:
      callback.on_training_start()

  def on_training_end(self):
    """Method called after training has finished.

    Can be used for additional logging or similar.
    """
    logging.info("Finished training")
    for callback in self.callbacks:
      callback.on_training_end()

  def on_training_epoch_start(self, epoch_idx: int):
    """Method called at the start of each training epoch. Can be used for additional logging or
    similar.

    Args:
        epoch_idx: Index of the training epoch that has started.
    """
    logging.info(f"Starting training epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_training_epoch_start(epoch_idx)

  def on_training_epoch_end(self, train_metrics: dict[str, Any], epoch_idx: int):
    """Method called at the end of each training epoch. Can be used for additional logging or
    similar.

    Args:
        epoch_idx: Index of the training epoch that has finished.
    """
    logging.info(f"Finished training epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_training_epoch_end(train_metrics, epoch_idx)

  def on_validation_epoch_start(self, epoch_idx: int):
    """Method called at the start of each validation epoch. Can be used for additional logging
    or similar.

    Args:
        epoch_idx: Index of the training epoch at which validation was started.
    """
    logging.info(f"Starting validation epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_validation_epoch_start(epoch_idx)

  def on_validation_epoch_end(self, eval_metrics: dict[str, Any], epoch_idx: int):
    """Method called at the end of each validation epoch. Can be used for additional logging
    and evaluation.

    Args:
        epoch_idx: Index of the training epoch at which validation was performed.
        eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
        val_loader: Data loader of the validation set, to support additional
            evaluation.
    """
    logging.info(f"Finished validation epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_validation_epoch_end(eval_metrics, epoch_idx)
    if (
      self.checkpoint_manager is not None
      and epoch_idx % self.trainer_config.checkpoint_config.every_n_epochs == 0
    ):
      self.checkpoint_manager.save_model(eval_metrics, epoch_idx)

  def on_test_epoch_start(self, epoch_idx: int):
    """Method called at the start of each test epoch. Can be used for additional logging or
    similar.

    Args:
        epoch_idx: Index of the training epoch at which testing was started.
    """
    logging.info(f"Starting test epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_test_epoch_start(epoch_idx)

  def on_test_epoch_end(self, test_metrics: dict[str, Any], epoch_idx: int):
    """Method called at the end of each test epoch.

    Can be used for additional logging and evaluation.

    Args:
        epoch_idx: Index of the training epoch at which testing was performed.
        test_metrics: A dictionary of the test metrics. New metrics added to
            this dictionary will be logged as well.
        test_loader: Data loader of the test set, to support additional
            evaluation.
    """
    logging.info(f"Finished test epoch {epoch_idx}")
    for callback in self.callbacks:
      callback.on_test_epoch_end(test_metrics, epoch_idx)

  def load_model(self, epoch_idx: int = -1, raise_if_not_found: bool = True):
    """Loads model parameters and batch statistics from the logging directory."""
    logging.info(f"Loading model from epoch {epoch_idx}")
    state_dict = None
    for callback in self.callbacks:
      if isinstance(callback, ModelCheckpoint):
        state_dict = callback.load_model(epoch_idx)
        break
    if state_dict is None:
      if raise_if_not_found:
        raise ValueError("No model checkpoint callback found in callbacks.")
      else:
        logging.warning("No model checkpoint callback found in callbacks.")
    else:
      self.restore(state_dict)

  def restore(self, state_dict: dict[str, Any]):
    """Restores the state of the trainer from a state dictionary.

    Args:
        state_dict: State dictionary to restore from.
    """
    logging.info("Restoring trainer state with keys " + str(state_dict.keys()))
    state_dict.pop("metrics")
    state_dict.pop("metadata")
    print("Restore state")
    self.state = TrainState.create(
      apply_fn=self.model.apply,
      # Optimizer will be overwritten when training starts
      tx=self.state.tx if self.state.tx else optax.sgd(0.1),
      rng=self.state.rng,
      **state_dict,
    )

  def bind_model(self):
    """Returns a model with parameters bound to it. Enables an easier inference access.

    Returns:
        The model with parameters and evt. batch statistics bound to it.
    """
    params = {"params": self.state.params}
    if self.state.mutable_variables is not None:
      params.update(self.state.mutable_variables)
    return self.model.bind(params)
