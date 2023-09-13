@keras_export("keras.metrics.Metric")
class Metric(base_layer.Layer, metaclass=abc.ABCMeta):
    """Encapsulates metric logic and state.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: Additional layer keywords arguments.
    Standalone usage:
    ```python
    m = SomeMetric(...)
    for input in ...:
      m.update_state(input)
    print('Final result: ', m.result().numpy())
    ```
    Usage with `compile()` API:
    ```python
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    model.fit(dataset, epochs=10)
    ```
    To be implemented by subclasses:
    * `__init__()`: All state variables should be created in this method by
      calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
      self.var.assign_add(...).
    * `result()`: Computes and returns a scalar value or a dict of scalar values
      for the metric from the state variables.
    Example subclass implementation:
    ```python
    class BinaryTruePositives(tf.keras.metrics.Metric):
      def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          sample_weight = tf.broadcast_to(sample_weight, values.shape)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))
      def result(self):
        return self.true_positives
    ```
    """
    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.stateful = True  # All metric layers are stateful.
        self.built = True
        if not base_layer_utils.v2_dtype_behavior_enabled():
            # We only do this when the V2 behavior is not enabled, as when it is
            # enabled, the dtype already defaults to floatx.
            self._dtype = (
                backend.floatx() if dtype is None else tf.as_dtype(dtype).name
            )
    def __new__(cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)
        # If `update_state` is not in eager/tf.function and it is not from a
        # built-in metric, wrap it in `tf.function`. This is so that users
        # writing custom metrics in v1 need not worry about control dependencies
        # and return ops.
        if base_layer_utils.is_in_eager_or_tf_function() or is_built_in(cls):
            obj_update_state = obj.update_state
            def update_state_fn(*args, **kwargs):
                control_status = tf.__internal__.autograph.control_status_ctx()
                ag_update_state = tf.__internal__.autograph.tf_convert(
                    obj_update_state, control_status
                )
                return ag_update_state(*args, **kwargs)
        else:
            if isinstance(obj.update_state, tf.__internal__.function.Function):
                update_state_fn = obj.update_state
            else:
                update_state_fn = tf.function(obj.update_state)
        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), obj
        )
        obj_result = obj.result
        def result_fn(*args, **kwargs):
            control_status = tf.__internal__.autograph.control_status_ctx()
            ag_result = tf.__internal__.autograph.tf_convert(
                obj_result, control_status
            )
            return ag_result(*args, **kwargs)
        obj.result = types.MethodType(
            metrics_utils.result_wrapper(result_fn), obj
        )
        return obj
    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value.
        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric,
            passed on to `update_state()`.
        Returns:
          The metric value tensor.
        """
        def replica_local_fn(*args, **kwargs):
            """Updates the state of the metric in a replica-local context."""
            if any(
                isinstance(arg, keras_tensor.KerasTensor)
                for arg in tf.nest.flatten((args, kwargs))
            ):
                update_op = None
            else:
                update_op = self.update_state(*args, **kwargs)
            update_ops = []
            if update_op is not None:
                update_ops.append(update_op)
            with tf.control_dependencies(update_ops):
                result_t = self.result()
                # If the metric object return a dictionary as a result, wrap it
                # with our custom dict object so we can attach the metric object
                # to it.
                if isinstance(result_t, dict):
                    result_t = _MetricDict(**result_t)
                # We are adding the metric object as metadata on the result
                # tensor.  This is required when we want to use a metric with
                # `add_metric` API on a Model/Layer in graph mode. This metric
                # instance will later be used to reset variable state after each
                # epoch of training.
                # Example:
                #   model = Model()
                #   mean = Mean()
                #   model.add_metric(mean(values), name='mean')
                result_t._metric_obj = self
                return result_t
        from keras.distribute import (
            distributed_training_utils,
        )
        return distributed_training_utils.call_replica_local_fn(
            replica_local_fn, *args, **kwargs
        )
    def __str__(self):
        args = ",".join(f"{k}={v}" for k, v in self.get_config().items())
        return f"{self.__class__.__name__}({args})"
    def __deepcopy__(self, memo=None):
        try:
            new_self = self.from_config(self.get_config())
        except NotImplementedError as e:
            raise NotImplementedError(
                "Calling `__deepcopy__()` on a Keras metric "
                "requires the metric to be serializable,  "
                "i.e. it should implement `get_config()`.\n\n"
                f"Error encountered during serialization: [{e}]"
            )
        # Note that metrics don't implement `build()` so their variables
        # are readily available after instantiation.
        if self.weights:
            new_self.set_weights(self.get_weights())
        memo[self] = new_self
        return new_self
    @property
    def dtype(self):
        return self._dtype
    def get_config(self):
        """Returns the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype}
    def reset_state(self):
        """Resets all of the metric state variables.
        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        if not generic_utils.is_default(self.reset_states):
            warnings.warn(
                "Metric %s implements a `reset_states()` method; rename it "
                'to `reset_state()` (without the final "s"). The name '
                "`reset_states()` has been deprecated to improve API "
                "consistency." % (self.__class__.__name__,),
                stacklevel=2,
            )
            return self.reset_states()
        else:
            backend.batch_set_value([(v, 0) for v in self.variables])
    @abc.abstractmethod
    def update_state(self, *args, **kwargs):
        """Accumulates statistics for the metric.
        Note: This function is executed as a graph function in graph mode.
        This means:
          a) Operations on the same resource are executed in textual order.
             This should make it easier to do things like add the updated
             value of a variable to another, for example.
          b) You don't need to worry about collecting the update ops to execute.
             All update ops added to the graph by this function will be
             executed.
          As a result, code should generally work the same way with graph or
          eager execution.
        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric.
        """
        raise NotImplementedError("Must be implemented in subclasses.")
    def merge_state(self, metrics):
        """Merges the state from one or more metrics.
        This method can be used by distributed systems to merge the state
        computed by different metric instances. Typically the state will be
        stored in the form of the metric's weights. For example, a
        tf.keras.metrics.Mean metric contains a list of two weight values: a
        total and a count. If there were two instances of a
        tf.keras.metrics.Accuracy that each independently aggregated partial
        state for an overall accuracy calculation, these two metric's states
        could be combined as follows:
        >>> m1 = tf.keras.metrics.Accuracy()
        >>> _ = m1.update_state([[1], [2]], [[0], [2]])
        >>> m2 = tf.keras.metrics.Accuracy()
        >>> _ = m2.update_state([[3], [4]], [[3], [4]])
        >>> m2.merge_state([m1])
        >>> m2.result().numpy()
        0.75
        Args:
          metrics: an iterable of metrics. The metrics must have compatible
            state.
        Raises:
          ValueError: If the provided iterable does not contain metrics matching
            the metric's required specifications.
        """
        assign_add_ops = []
        for metric in metrics:
            if len(self.weights) != len(metric.weights):
                raise ValueError(
                    f"Metric {metric} is not compatible with {self}"
                )
            for weight, weight_to_add in zip(self.weights, metric.weights):
                assign_add_ops.append(weight.assign_add(weight_to_add))
        return assign_add_ops
    @abc.abstractmethod
    def result(self):
        """Computes and returns the scalar metric value tensor or a dict of
        scalars.
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        Returns:
          A scalar tensor, or a dictionary of scalar tensors.
        """
        raise NotImplementedError("Must be implemented in subclasses.")
    ### For use by subclasses ###
    @doc_controls.for_subclass_implementers
    def add_weight(
        self,
        name,
        shape=(),
        aggregation=tf.VariableAggregation.SUM,
        synchronization=tf.VariableSynchronization.ON_READ,
        initializer=None,
        dtype=None,
    ):
        """Adds state variable. Only for use by subclasses."""
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
        else:
            strategy = None
        additional_kwargs = {}
        # TODO(b/120571621): Make `ON_READ` work with Keras metrics on TPU.
        if backend.is_tpu_strategy(strategy):
            synchronization = tf.VariableSynchronization.ON_WRITE
        if getattr(self, "_mesh", None) is not None:
            # When self._mesh is set, it means this metric is used for DTensor.
            additional_kwargs = {
                "layout": dtensor.Layout.replicated(
                    self._mesh, tf.TensorShape(shape).rank
                )
            }
        if tf_utils.in_local_vars_context():
            # Metrics created within a remotely-executed tf.function during
            # parameter server evaluation should use tf2 Variables, so that they
            # can be local variables that are freely usable and mutable within
            # the function, using the
            # `experimental_enable_variable_lifting=False` argument. This
            # supports a visitation guarantee for model evaluation.
            def local_v2_var_creator(
                initializer=None, dtype=None, shape=None, **kwargs
            ):
                init_val, var_dtype = base_layer_utils.infer_init_val_and_dtype(
                    initializer, dtype, shape
                )
                v1_only_args = ["use_resource", "collections"]
                for v1_arg in v1_only_args:
                    kwargs.pop(v1_arg, None)
                kwargs["experimental_enable_variable_lifting"] = False
                return tf.Variable(
                    initial_value=init_val,
                    dtype=var_dtype,
                    shape=shape,
                    **kwargs,
                )
            additional_kwargs["getter"] = local_v2_var_creator
        with tf_utils.maybe_init_scope(layer=self):
            return super().add_weight(
                name=name,
                shape=shape,
                dtype=self._dtype if dtype is None else dtype,
                trainable=False,
                initializer=initializer,
                collections=[],
                synchronization=synchronization,
                aggregation=aggregation,
                **additional_kwargs,
            )
    ### End: For use by subclasses ###
    @property
    def trainable_weights(self):
        # Overridden from Layer class to track submetric weights.
        if self.trainable:
            trainable_weights = self._trainable_weights
            for m in self._metrics:
                trainable_weights += m.trainable_weights
            return self._dedup_weights(trainable_weights)
        else:
            return []
    @property
    def non_trainable_weights(self):
        # Overridden from Layer class to track submetric weights.
        if self.trainable:
            non_trainable_weights = self._non_trainable_weights
            for m in self._metrics:
                non_trainable_weights += m.non_trainable_weights
        else:
            non_trainable_weights = (
                self._non_trainable_weights + self._trainable_weights
            )
            for m in self._metrics:
                non_trainable_weights += m.weights
        return self._dedup_weights(non_trainable_weights)
    @property
    def _trackable_saved_model_saver(self):
        return metric_serialization.MetricSavedModelSaver(self)
    @generic_utils.default
    @doc_controls.do_not_generate_docs
    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        return self.reset_state()
