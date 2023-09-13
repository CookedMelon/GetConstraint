"/home/cc/Workspace/tfconstraint/keras/optimizers/optimizer.py"
@keras_export(
    "keras.optimizers.Optimizer",
    "keras.optimizers.experimental.Optimizer",
    v1=[],
)
class Optimizer(_BaseOptimizer):
    """Abstract optimizer base class.
    This class supports distributed training. If you want to implement your own
    optimizer, please subclass this class instead of _BaseOptimizer.
    Args:
      {{base_optimizer_keyword_args}}
    ### Usage
    ```python
    # Create an optimizer with the desired parameters.
    opt = keras.optimizers.SGD(learning_rate=0.1)
    var1, var2 = tf.Variable(1.0), tf.Variable(2.0)
    # `loss` is a callable that takes no argument and returns the value
    # to minimize.
    loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
    # Call minimize to update the list of variables.
    opt.minimize(loss, var_list=[var1, var2])
    ```
    ### Processing gradients before applying them
    Calling `minimize()` takes care of both computing the gradients and
    applying them to the variables. If you want to process the gradients
    before applying them you can instead use the optimizer in three steps:
    1.  Compute the gradients with `tf.GradientTape`.
    2.  Process the gradients as you wish.
    3.  Apply the processed gradients with `apply_gradients()`.
    Example:
    ```python
    # Create an optimizer.
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
    var1, var2 = tf.Variable(1.0), tf.Variable(2.0)
    # Compute the gradients for a list of variables.
    with tf.GradientTape() as tape:
      loss = 3 * var1 * var1 + 2 * var2 * var2
    grads = tape.gradient(loss, [var1, var2])
    # Process the gradients.
    grads[0] = grads[0] + 1
    # Ask the optimizer to apply the gradients on variables.
    opt.apply_gradients(zip(grads, [var1, var2]))
    ```
    ### Dynamic learning rate
    Dynamic learning rate can be achieved by setting learning rate as a built-in
    or customized `tf.keras.optimizers.schedules.LearningRateSchedule`.
    Example:
    >>> var = tf.Variable(np.random.random(size=(1,)))
    >>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    ...   initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
    >>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)
    >>> loss = lambda: 3 * var
    >>> opt.minimize(loss, var_list=[var])
    ### Gradients clipping
    Users can clip the gradients before applying to variables by setting
    `clipnorm`, `clipvalue` and `global_clipnorm`. Notice that `clipnorm` and
    `global_clipnorm` can only have one being set.
    Example:
    >>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=1, clipvalue=1)
    >>> var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
    >>> with tf.GradientTape() as tape:
    ...   loss = 2 * var1 + 2 * var2
    >>> grads = tape.gradient(loss, [var1, var2])
    >>> print([grads[0].numpy(), grads[1].numpy()])
    [2.0, 2.0]
    >>> opt.apply_gradients(zip(grads, [var1, var2]))
    >>> # Without clipping, we should get [0, 0], but as gradients are clipped
    >>> # to have max value 1, we get [1.0, 1.0].
    >>> print([var1.numpy(), var2.numpy()])
    [1.0, 1.0]
    ### Using weight decay.
    Weight decay in certain scenarios can boost the model's performance. Keras
    has built-in support for weight decay in all optimizers. Users can apply
    weight decay by setting `weight_decay` argument.
    >>> opt = tf.keras.optimizers.experimental.SGD(1, weight_decay=0.004)
    >>> grads, var1, var2 = tf.zeros(()), tf.Variable(2.0), tf.Variable(2.0)
    >>> # You can exclude variables from weight decay, in this case we
    >>> # exclude `var2`.
    >>> opt.exclude_from_weight_decay(var_list=[var2])
    >>> opt.apply_gradients(zip([grads, grads], [var1, var2]))
    >>> print([var1.numpy(), var2.numpy()])
    [1.992, 2.0]
    ### Using exponential moving average.
    Empirically it has been found that using the exponential moving average
    (EMA) of the trained parameters of a deep network achieves a better
    performance than using its trained parameters directly. Keras optimizers
    allows users to compute this moving average and overwrite the model
    variables at desired time.
    Example:
    ```python
    # Create an SGD optimizer with EMA on. `ema_momentum` controls the decay
    # rate of the moving average. `ema_momentum=1` means no decay and the stored
    # moving average is always model variable's initial value before training.
    # Reversely, `ema_momentum=0` is equivalent to not using EMA.
    # `ema_overwrite_frequency=3` means every 3 iterations, we overwrite the
    # trainable variables with their moving average values.
    opt = tf.keras.optimizers.experimental.SGD(
        learning_rate=1,
        use_ema=True,
        ema_momentum=0.5,
        ema_overwrite_frequency=3)
    var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
    with tf.GradientTape() as tape:
      loss = var1 + var2
    grads = tape.gradient(loss, [var1, var2])
    # First iteration: [var1, var2] = [1.0, 1.0]
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])
    # Second iteration: [var1, var2] = [0.0, 0.0]
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])
    # Third iteration, without EMA, we should see [var1, var2] = [-1.0, -1.0],
    # but overwriting results in [var1, var2] = [-0.125, -0.125]. The full
    # calculation for the moving average of var1 is:
    # var1=2*0.5**3+1*(1-0.5)*0.5**2+0*(1-0.5)*0.5**1+(-1)*(1-0.5)=-0.125.
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])
    ```
    When optimizer is constructed with `use_ema=True`, in custom training loop,
    users can explicitly call `finalize_variable_values()` to overwrite
    trainable variables with their EMA values. `finalize_variable_values()` is
    by default called at the end of `model.fit()`.
    ### Use with `tf.distribute.Strategy`
    This optimizer class is `tf.distribute.Strategy` aware, which means it
    automatically sums gradients across all replicas. To aggregate gradients
    yourself, call `apply_gradients` with `skip_aggregate_gradients` set to
    True.  This is useful if you need to process aggregated gradients.
    ```python
    # This example is not runnable, it consists of dummy code for simple
    # tutorial.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      opt = tf.keras.optimizers.experimental.SGD()
      model = magic_function_that_returns_model()
      gradients = magic_function_that_returns_gradients()
      # Custom logic to aggregate gradients.
      gradients = strategy.reduce("SUM", gradients, axis=None)
      opt.apply_gradients(zip(gradients, model.trainable_variables),
          skip_aggregate_gradients=True)
    ```
    ### Creating a custom optimizer
    If you intend to create your own optimization algorithm, please inherit from
    this class and override the following methods:
      - `build`: Create your optimizer-related variables, such as `momentums` in
        SGD optimizer.
      - `update_step`: Implement your optimizer's updating logic.
      - `get_config`: serialization of the optimizer, include all hyper
        parameters.
    Your optimizer would automatically be compatible with tensorflow distributed
    training if you subclass `optimizer_experimental.Optimizer`.
    """
    def __init__(
        self,
        name,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        **kwargs,
    ):
        """Create a new Optimizer."""
        mesh = kwargs.pop("mesh", None)
        self._mesh = mesh
        super().__init__(
            name,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            ema_overwrite_frequency,
            jit_compile,
            **kwargs,
        )
        self._distribution_strategy = tf.distribute.get_strategy()
        self._run_with_dtensor = dtensor_utils.running_with_dtensor_strategy()
    def add_variable_from_reference(
        self, model_variable, variable_name, shape=None, initial_value=None
    ):
        if self._mesh:
            if initial_value is None:
                # Use tf.zeros_like which will propagate the layout information
                # from the model weights if any.
                initial_value = tf.zeros_like(model_variable)
            elif isinstance(initial_value, tf.Tensor):
                initial_value = tf.experimental.dtensor.copy_to_mesh(
                    initial_value,
                    tf.experimental.dtensor.Layout.replicated(
                        self._mesh, rank=initial_value.shape.rank
                    ),
                )
            variable = tf.experimental.dtensor.DVariable(
                initial_value=initial_value,
                name=f"{variable_name}/{model_variable._shared_name}",
                dtype=model_variable.dtype,
                trainable=False,
            )
            self._variables.append(variable)
            return variable
        else:
            strategy = tf.distribute.get_strategy()
            with strategy.extended.colocate_vars_with(model_variable):
                return super().add_variable_from_reference(
                    model_variable, variable_name, shape, initial_value
                )
    def _create_iteration_variable(self):
        if self._mesh:
            init_val = tf.constant(0, dtype=tf.int64)
            init_val = tf.experimental.dtensor.copy_to_mesh(
                init_val,
                tf.experimental.dtensor.Layout.replicated(self._mesh, rank=0),
            )
            with tf.init_scope():
                # Lift the variable creation to init scope to avoid environment
                # issue.
                self._iterations = tf.experimental.dtensor.DVariable(
                    init_val, name="iteration"
                )
            self._variables.append(self._iterations)
        else:
            super()._create_iteration_variable()
    def _var_key(self, variable):
        """Get a unique identifier of the given variable."""
        # Get the distributed variable if it exists.
        # TODO(b/197554203): replace _distributed_container() with a public api.
        if hasattr(variable, "_distributed_container"):
            variable = variable._distributed_container()
        elif (
            tf_utils.is_extension_type(variable)
            and hasattr(variable, "handle")
            and hasattr(variable.handle, "_distributed_container")
        ):
            # For ResourceVariables, the _distributed_container attribute
            # is added to their handle tensors.
            variable = variable.handle._distributed_container()
        return super()._var_key(variable)
    def aggregate_gradients(self, grads_and_vars):
        """Aggregate gradients on all devices.
        By default, we will perform reduce_sum of gradients across devices.
        Users can implement their own aggregation logic by overriding this
        method.
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
        Returns:
          List of (gradient, variable) pairs.
        """
        if self._mesh or self._run_with_dtensor:
            raise NotImplementedError(
                "Dtensor doesn't need to manually aggregate gradients"
            )
        else:
            return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)
    def apply_gradients(
        self,
        grads_and_vars,
        name=None,
        skip_gradients_aggregation=False,
        **kwargs,
    ):
        """Apply gradients to variables.
        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.
          skip_gradients_aggregation: If true, gradients aggregation will not be
            performed inside optimizer. Usually this arg is set to True when you
            write custom code aggregating gradients outside the optimizer.
          **kwargs: keyword arguments only used for backward compatibility.
        Returns:
          A `tf.Variable`, representing the current iteration.
        Raises:
          TypeError: If `grads_and_vars` is malformed.
          RuntimeError: If called in a cross-replica context.
        """
        if self._mesh or self._run_with_dtensor:
            # Skip any usage of strategy logic for DTensor
            return super().apply_gradients(grads_and_vars, name=name)
        # `experimental_aggregate_gradients` is an arg in `apply_gradients` of
        # v2 optimizer -- the reverse of `skip_gradients_aggregation`.
        # We read it from kwargs for backward compatibility.
        experimental_aggregate_gradients = kwargs.pop(
            "experimental_aggregate_gradients", True
        )
        if not skip_gradients_aggregation and experimental_aggregate_gradients:
            grads_and_vars = self.aggregate_gradients(grads_and_vars)
        return super().apply_gradients(grads_and_vars, name=name)
    def _apply_weight_decay(self, variables):
        # Apply weight decay in distributed setup.
        if self.weight_decay is None:
            return
        def distributed_apply_weight_decay(distribution, variables, **kwargs):
            def weight_decay_fn(variable):
                if self._use_weight_decay(variable):
                    lr = tf.cast(self.learning_rate, variable.dtype)
                    wd = tf.cast(self.weight_decay, variable.dtype)
                    variable.assign_sub(variable * wd * lr)
            for variable in variables:
                distribution.extended.update(
                    variable, weight_decay_fn, group=False
                )
        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay,
            self._distribution_strategy,
            variables,
        )
    def _internal_apply_gradients(self, grads_and_vars):
        if self._mesh or self._run_with_dtensor:
            # Skip any usage of strategy logic for DTensor
            return super()._internal_apply_gradients(grads_and_vars)
        return tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_apply_gradients_fn,
            self._distribution_strategy,
            grads_and_vars,
        )
    def _overwrite_model_variables_with_average_value_helper(self, var_list):
        """Helper function to _overwrite_model_variables_with_average_value.
        This function overwrites variables on each device.
        Args:
          var_list: list of model variables.
        """
        if self._mesh or self._run_with_dtensor:
            # Skip any usage of strategy logic for DTensor
            super()._overwrite_model_variables_with_average_value_helper(
                var_list
            )
        strategy = self._distribution_strategy
        # Override model variable by the stored average value on all devices.
        for var, average_var in zip(
            var_list, self._model_variables_moving_average
        ):
            strategy.extended.update(
                var, lambda a, b: a.assign(b), args=(average_var,)
            )
    def _build_learning_rate(self, learning_rate):
        if not self._mesh:
            return super()._build_learning_rate(learning_rate)
        # For DTensor
        variable_creation = tf.experimental.dtensor.DVariable
        init_value_convert_fn = lambda x: tf.experimental.dtensor.copy_to_mesh(
            x, tf.experimental.dtensor.Layout.replicated(self._mesh, rank=0)
        )
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            current_learning_rate = tf.convert_to_tensor(
                learning_rate(self.iterations)
            )
            current_learning_rate = init_value_convert_fn(current_learning_rate)
            # Create a variable to hold the current learning rate.
            # Note that the init value `learning_rate(self.iterations)` should
            # have the correct layout information from self.iterations.
            self._current_learning_rate = variable_creation(
                current_learning_rate,
                name="learning_rate",
                dtype=tf.float32,
            )
            return learning_rate
        init_val = init_value_convert_fn(
            tf.constant(learning_rate, dtype=tf.float32)
        )
        return variable_creation(
            init_val,
            name="learning_rate",
            dtype=backend.floatx(),
            trainable=False,
        )
    def _update_model_variables_moving_average(self, var_list):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            def update_average(average, var):
                average.assign(
                    self.ema_momentum * average + (1 - self.ema_momentum) * var
                )
            for var, average in zip(
                var_list, self._model_variables_moving_average
            ):
                self._distribution_strategy.extended.update(
                    average, update_average, args=(var,), group=False
                )
    def _distributed_apply_gradients_fn(
        self, distribution, grads_and_vars, **kwargs
    ):
        """`apply_gradients` using a `DistributionStrategy`."""
        def apply_grad_to_update_var(var, grad):
            if self.jit_compile:
                return self._update_step_xla(grad, var, id(self._var_key(var)))
            else:
                return self._update_step(grad, var)
        for grad, var in grads_and_vars:
            distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False
            )
        if self.use_ema:
            _, var_list = zip(*grads_and_vars)
            self._update_model_variables_moving_average(var_list)
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                tf.cond(
                    tf.cast(should_overwrite_model_vars, tf.bool),
                    true_fn=lambda: self._overwrite_model_variables_with_average_value(  # noqa: E501
                        var_list
                    ),
                    false_fn=lambda: None,
                )
        return self.iterations.assign_add(1)
class RestoredOptimizer(Optimizer):
    def __init__(self):
        super().__init__("RestoredOptimizer")
    def get_config(self):
        raise NotImplementedError(
            "Restoring functional Optimizers from SavedModels is not currently "
            "supported. Please file a feature request if this limitation "
            "bothers you."
        )
class CallableList(list):
    """Temporary shim to support both `opt.variables()` and `opt.variables`."""
    def __call__(self):
        return self
# Register the optimizer for loading from saved_model purpose.
tf.__internal__.saved_model.load.register_revived_type(
    "experimentalOptimizer",
    lambda obj: isinstance(obj, Optimizer),
    versions=[
        tf.__internal__.saved_model.load.VersionedTypeRegistration(
            object_factory=lambda proto: RestoredOptimizer(),
            version=2,
            min_producer_version=1,
            min_consumer_version=1,
        )
    ],
)
Optimizer.__doc__ = Optimizer.__doc__.replace(
    "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args
)
