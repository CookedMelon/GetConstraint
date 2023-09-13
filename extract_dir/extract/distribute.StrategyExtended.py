@tf_export("distribute.StrategyExtended", v1=[])
class StrategyExtendedV2(object):
  """Additional APIs for algorithms that need to be distribution-aware.
  Note: For most usage of `tf.distribute.Strategy`, there should be no need to
  call these methods, since TensorFlow libraries (such as optimizers) already
  call these methods when needed on your behalf.
  Some common use cases of functions on this page:
  * _Locality_
  `tf.distribute.DistributedValues` can have the same _locality_ as a
  _distributed variable_, which leads to a mirrored value residing on the same
  devices as the variable (as opposed to the compute devices). Such values may
  be passed to a call to `tf.distribute.StrategyExtended.update` to update the
  value of a variable. You may use
  `tf.distribute.StrategyExtended.colocate_vars_with` to give a variable the
  same locality as another variable. You may convert a "PerReplica" value to a
  variable's locality by using `tf.distribute.StrategyExtended.reduce_to` or
  `tf.distribute.StrategyExtended.batch_reduce_to`.
  * _How to update a distributed variable_
  A distributed variable is variables created on multiple devices. As discussed
  in the [glossary](https://www.tensorflow.org/api_docs/python/tf/distribute),
  mirrored variable and SyncOnRead variable are two examples. The standard
  pattern for updating distributed variables is to:
  1. In your function passed to `tf.distribute.Strategy.run`,
     compute a list of (update, variable) pairs. For example, the update might
     be a gradient of the loss with respect to the variable.
  2. Switch to cross-replica mode by calling
     `tf.distribute.get_replica_context().merge_call()` with the updates and
     variables as arguments.
  3. Call
     `tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)`
     (for one variable) or `tf.distribute.StrategyExtended.batch_reduce_to`
     (for a list of variables) to sum the updates.
  4. Call `tf.distribute.StrategyExtended.update(v)` for each variable to update
     its value.
  Steps 2 through 4 are done automatically by class
  `tf.keras.optimizers.Optimizer` if you call its
  `tf.keras.optimizers.Optimizer.apply_gradients` method in a replica context.
  In fact, a higher-level solution to update a distributed variable is by
  calling `assign` on the variable as you would do to a regular `tf.Variable`.
  You can call the method in both _replica context_ and _cross-replica context_.
  For a _mirrored variable_, calling `assign` in _replica context_ requires you
  to specify the `aggregation` type in the variable constructor. In that case,
  the context switching and sync described in steps 2 through 4 are handled for
  you. If you call `assign` on _mirrored variable_ in _cross-replica context_,
  you can only assign a single value or assign values from another mirrored
  variable or a mirrored `tf.distribute.DistributedValues`. For a _SyncOnRead
  variable_, in _replica context_, you can simply call `assign` on it and no
  aggregation happens under the hood. In _cross-replica context_, you can only
  assign a single value to a SyncOnRead variable. One example case is restoring
  from a checkpoint: if the `aggregation` type of the variable is
  `tf.VariableAggregation.SUM`, it is assumed that replica values were added
  before checkpointing, so at the time of restoring, the value is divided by
  the number of replicas and then assigned to each replica; if the `aggregation`
  type is `tf.VariableAggregation.MEAN`, the value is assigned to each replica
  directly.
  """
  def __init__(self, container_strategy):
    self._container_strategy_weakref = weakref.ref(container_strategy)
    self._default_device = None
    # This property is used to determine if we should set drop_remainder=True
    # when creating Datasets from numpy array inputs.
    self._require_static_shapes = False
  def _resource_creator_scope(self):
    """Returns one or a list of ops.resource_creator_scope for some Strategy."""
    return None
  def _container_strategy(self):
    """Get the containing `tf.distribute.Strategy`.
    This should not generally be needed except when creating a new
    `ReplicaContext` and to validate that the caller is in the correct
    `scope()`.
    Returns:
      The `tf.distribute.Strategy` such that `strategy.extended` is `self`.
    """
    container_strategy = self._container_strategy_weakref()
    assert container_strategy is not None
    return container_strategy
  def _scope(self, strategy):
    """Implementation of tf.distribute.Strategy.scope()."""
    def creator_with_resource_vars(next_creator, **kwargs):
      """Variable creator to use in `_CurrentDistributionContext`."""
      if ops.inside_function():
        if_graph_building = "graph_building"
      else:
        if_graph_building = "not_graph_building"
      with monitoring.MonitoredTimer(distributed_variable_creation_time_counter.get_cell(strategy.__class__.__name__, if_graph_building)):
        _require_strategy_scope_extended(self)
        kwargs["use_resource"] = True
        kwargs["distribute_strategy"] = strategy
        # Unwrap `initial_value` if it is a `CheckpointInitialValue` to avoid
        # dereferencing a `Tensor` that is without a `name`. We still need to
        # propagate the metadata it's holding.
        if isinstance(kwargs["initial_value"], trackable.CheckpointInitialValue):
          checkpoint_restore_uid = kwargs[
              "initial_value"].checkpoint_position.restore_uid
          kwargs["initial_value"] = kwargs["initial_value"].wrapped_value
        elif isinstance(kwargs["initial_value"],
                        trackable.CheckpointInitialValueCallable):
          checkpoint_restore_uid = kwargs[
              "initial_value"].checkpoint_position.restore_uid
        elif (isinstance(kwargs["initial_value"], functools.partial) and
              isinstance(kwargs["initial_value"].func,
                         trackable.CheckpointInitialValueCallable)):
          # Some libraries (e.g, Keras) create partial function out of initializer
          # to bind shape/dtype, for example:
          #  initial_val = functools.partial(initializer, shape, dtype=dtype)
          # Therefore to get the restore_uid we need to examine the "func" of
          # the partial function.
          checkpoint_restore_uid = kwargs[
              "initial_value"].func.checkpoint_position.restore_uid
        else:
          checkpoint_restore_uid = None
        created = self._create_variable(next_creator, **kwargs)
        if checkpoint_restore_uid is not None:
          # pylint: disable=protected-access
          # Let the checkpointing infrastructure know that the variable was
          # already restored so it doesn't waste memory loading the value again.
          # In this case of CheckpointInitialValueCallable this may already be
          # done by the final variable creator, but it doesn't hurt to do it
          # again.
          created._maybe_initialize_trackable()
          created._update_uid = checkpoint_restore_uid
          # pylint: enable=protected-access
        return created
    def distributed_getter(getter, *args, **kwargs):
      if not self._allow_variable_partition():
        if kwargs.pop("partitioner", None) is not None:
          tf_logging.log_first_n(
              tf_logging.WARN, "Partitioned variables are disabled when using "
              "current tf.distribute.Strategy.", 1)
      return getter(*args, **kwargs)
    return _CurrentDistributionContext(
        strategy,
        variable_scope.variable_creator_scope(creator_with_resource_vars),
        variable_scope.variable_scope(
            variable_scope.get_variable_scope(),
            custom_getter=distributed_getter),
        strategy.extended._resource_creator_scope(),  # pylint: disable=protected-access
        self._default_device)
  def _allow_variable_partition(self):
    return False
  def _create_variable(self, next_creator, **kwargs):
    # Note: should support "colocate_with" argument.
    raise NotImplementedError("must be implemented in descendants")
  def variable_created_in_scope(self, v):
    """Tests whether `v` was created while this strategy scope was active.
    Variables created inside the strategy scope are "owned" by it:
    >>> strategy = tf.distribute.MirroredStrategy()
    >>> with strategy.scope():
    ...   v = tf.Variable(1.)
    >>> strategy.extended.variable_created_in_scope(v)
    True
    Variables created outside the strategy are not owned by it:
    >>> strategy = tf.distribute.MirroredStrategy()
    >>> v = tf.Variable(1.)
    >>> strategy.extended.variable_created_in_scope(v)
    False
    Args:
      v: A `tf.Variable` instance.
    Returns:
      True if `v` was created inside the scope, False if not.
    """
    return v._distribute_strategy == self._container_strategy_weakref()  # pylint: disable=protected-access
  def colocate_vars_with(self, colocate_with_variable):
    """Scope that controls which devices variables will be created on.
    No operations should be added to the graph inside this scope, it
    should only be used when creating variables (some implementations
    work by changing variable creation, others work by using a
    tf.compat.v1.colocate_with() scope).
    This may only be used inside `self.scope()`.
    Example usage:
    ```
    with strategy.scope():
      var1 = tf.Variable(...)
      with strategy.extended.colocate_vars_with(var1):
        # var2 and var3 will be created on the same device(s) as var1
        var2 = tf.Variable(...)
        var3 = tf.Variable(...)
      def fn(v1, v2, v3):
        # operates on v1 from var1, v2 from var2, and v3 from var3
      # `fn` runs on every device `var1` is on, `var2` and `var3` will be there
      # too.
      strategy.extended.update(var1, fn, args=(var2, var3))
    ```
    Args:
      colocate_with_variable: A variable created in this strategy's `scope()`.
        Variables created while in the returned context manager will be on the
        same set of devices as `colocate_with_variable`.
    Returns:
      A context manager.
    """
    def create_colocated_variable(next_creator, **kwargs):
      _require_strategy_scope_extended(self)
      kwargs["use_resource"] = True
      kwargs["colocate_with"] = colocate_with_variable
      return next_creator(**kwargs)
    _require_strategy_scope_extended(self)
    self._validate_colocate_with_variable(colocate_with_variable)
    return variable_scope.variable_creator_scope(create_colocated_variable)
  def _validate_colocate_with_variable(self, colocate_with_variable):
    """Validate `colocate_with_variable` argument to `colocate_vars_with`."""
    pass
  def _make_dataset_iterator(self, dataset):
    raise NotImplementedError("must be implemented in descendants")
  def _make_input_fn_iterator(self, input_fn, replication_mode):
    raise NotImplementedError("must be implemented in descendants")
  def _experimental_distribute_dataset(self, dataset, options):
    raise NotImplementedError("must be implemented in descendants")
  def _distribute_datasets_from_function(self, dataset_fn, options):
    raise NotImplementedError("must be implemented in descendants")
  def _experimental_distribute_values_from_function(self, value_fn):
    raise NotImplementedError("must be implemented in descendants")
  def _reduce(self, reduce_op, value):
    # Default implementation until we have an implementation for each strategy.
    dst = device_util.current() or self._default_device or "/device:CPU:0"
    return self._local_results(self.reduce_to(reduce_op, value, dst))[0]
  def reduce_to(self, reduce_op, value, destinations, options=None):
    """Combine (via e.g. sum or mean) values across replicas.
    `reduce_to` aggregates `tf.distribute.DistributedValues` and distributed
    variables. It supports both dense values and `tf.IndexedSlices`.
    This API currently can only be called in cross-replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.batch_reduce_to`: the batch version of
      this API.
    * `tf.distribute.ReplicaContext.all_reduce`: the counterpart of this API
      in replica context. It supports both batched and non-batched all-reduce.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.
    `destinations` specifies where to reduce the value to, e.g. "GPU:0". You can
    also pass in a `Tensor`, and the destinations will be the device of that
    tensor. For all-reduce, pass the same to `value` and `destinations`.
    It can be used in `tf.distribute.ReplicaContext.merge_call` to write code
    that works for all `tf.distribute.Strategy`.
    @tf.function
    def step_fn(var):
      def merge_fn(strategy, value, var):
        # All-reduce the value. Note that `value` here is a
        # `tf.distribute.DistributedValues`.
        reduced = strategy.extended.reduce_to(tf.distribute.ReduceOp.SUM,
            value, destinations=var)
        strategy.extended.update(var, lambda var, value: var.assign(value),
            args=(reduced,))
      value = tf.identity(1.)
      tf.distribute.get_replica_context().merge_call(merge_fn,
        args=(value, var))
    def run(strategy):
      with strategy.scope():
        v = tf.Variable(0.)
        strategy.run(step_fn, args=(v,))
        return v
    run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
    }
    run(tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    run(tf.distribute.OneDeviceStrategy("GPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value: a `tf.distribute.DistributedValues`, or a `tf.Tensor` like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.
    Returns:
      A tensor or value reduced to `destinations`.
    """
    if options is None:
      options = collective_util.Options()
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    assert (reduce_op == reduce_util.ReduceOp.SUM or
            reduce_op == reduce_util.ReduceOp.MEAN)
    return self._reduce_to(reduce_op, value, destinations, options)
  def _reduce_to(self, reduce_op, value, destinations, options):
    raise NotImplementedError("must be implemented in descendants")
  def batch_reduce_to(self, reduce_op, value_destination_pairs, options=None):
    """Combine multiple `reduce_to` calls into one for faster execution.
    Similar to `reduce_to`, but accepts a list of (value, destinations) pairs.
    It's more efficient than reduce each value separately.
    This API currently can only be called in cross-replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.reduce_to`: the non-batch version of
      this API.
    * `tf.distribute.ReplicaContext.all_reduce`: the counterpart of this API
      in replica context. It supports both batched and non-batched all-reduce.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.
    See `reduce_to` for more information.
    @tf.function
    def step_fn(var):
      def merge_fn(strategy, value, var):
        # All-reduce the value. Note that `value` here is a
        # `tf.distribute.DistributedValues`.
        reduced = strategy.extended.batch_reduce_to(
            tf.distribute.ReduceOp.SUM, [(value, var)])[0]
        strategy.extended.update(var, lambda var, value: var.assign(value),
            args=(reduced,))
      value = tf.identity(1.)
      tf.distribute.get_replica_context().merge_call(merge_fn,
        args=(value, var))
    def run(strategy):
      with strategy.scope():
        v = tf.Variable(0.)
        strategy.run(step_fn, args=(v,))
        return v
    run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
    }
    run(tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    run(tf.distribute.OneDeviceStrategy("GPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.Strategy.reduce_to` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.
    Returns:
      A list of reduced values, one per pair in `value_destination_pairs`.
    """
    if options is None:
      options = collective_util.Options()
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(reduce_op, variable_scope.VariableAggregation)
    if isinstance(reduce_op, six.string_types):
      reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    return self._batch_reduce_to(reduce_op, value_destination_pairs, options)
  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    return [
        self.reduce_to(reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]
  def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
    """All-reduce `value` across all replicas so that all get the final result.
    If `value` is a nested structure of tensors, all-reduces of these tensors
    will be batched when possible. `options` can be set to hint the batching
    behavior.
    This API must be called in a replica context.
    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: Value to be reduced. A tensor or a nested structure of tensors.
      options: A `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor.
    Returns:
      A tensor or a nested strucutre of tensors with the reduced values. The
      structure is the same as `value`.
    """
    if options is None:
      options = collective_util.Options()
    replica_context = get_replica_context()
    assert replica_context, (
        "`StrategyExtended._replica_ctx_all_reduce` must be called in"
        " a replica context")
    def merge_fn(_, flat_value):
      return self.batch_reduce_to(reduce_op, [(v, v) for v in flat_value],
                                  options)
    reduced = replica_context.merge_call(merge_fn, args=(nest.flatten(value),))
    return nest.pack_sequence_as(value, reduced)
  def _replica_ctx_update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` with `args` and `kwargs` to update `var`."""
    # This method is called by ReplicaContext.update. Strategies who'd like to
    # remove merge_call in this path should override this method.
    replica_context = get_replica_context()
    if not replica_context:
      raise ValueError("`StrategyExtended._replica_ctx_update` must be called "
                       "in a replica context.")
    def merge_fn(_, *merged_args, **merged_kwargs):
      return self.update(var, fn, merged_args, merged_kwargs, group=group)
    return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)
  def _gather_to(self, value, destinations, axis, options=None):
    """Gather `value` across replicas along axis-th dimension to `destinations`.
    `gather_to` gathers `tf.distribute.DistributedValues` or `tf.Tensor`-like
    object, along `axis`-th dimension. It supports only dense tensors but NOT
    sparse tensor. This API can only be called in cross-replica context.
    Args:
      value: a `tf.distribute.DistributedValues`, or a `tf.Tensor` like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.
    Returns:
      A tensor or value gathered to `destinations`.
    """
    _require_cross_replica_or_default_context_extended(self)
    assert not isinstance(destinations, (list, tuple))
    if options is None:
      options = collective_util.Options()
    return self._gather_to_implementation(value, destinations, axis, options)
  def _gather_to_implementation(self, value, destinations, axis, options):
    raise NotImplementedError("_gather_to must be implemented in descendants")
  def _batch_gather_to(self, value_destination_pairs, axis, options=None):
    _require_cross_replica_or_default_context_extended(self)
    if options is None:
      options = collective_util.Options()
    return [
        self._gather_to(t, destinations=v, axis=axis, options=options)
        for t, v in value_destination_pairs
    ]
  def update(self, var, fn, args=(), kwargs=None, group=True):
    """Run `fn` to update `var` using inputs mirrored to the same devices.
    `tf.distribute.StrategyExtended.update` takes a distributed variable `var`
    to be updated, an update function `fn`, and `args` and `kwargs` for `fn`. It
    applies `fn` to each component variable of `var` and passes corresponding
    values from `args` and `kwargs`. Neither `args` nor `kwargs` may contain
    per-replica values. If they contain mirrored values, they will be unwrapped
    before calling `fn`. For example, `fn` can be `assign_add` and `args` can be
    a mirrored DistributedValues where each component contains the value to be
    added to this mirrored variable `var`. Calling `update` will call
    `assign_add` on each component variable of `var` with the corresponding
    tensor value on that device.
    Example usage:
    ```python
    strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1']) # With 2
    devices
    with strategy.scope():
      v = tf.Variable(5.0, aggregation=tf.VariableAggregation.SUM)
    def update_fn(v):
      return v.assign(1.0)
    result = strategy.extended.update(v, update_fn)
    # result is
    # Mirrored:{
    #  0: tf.Tensor(1.0, shape=(), dtype=float32),
    #  1: tf.Tensor(1.0, shape=(), dtype=float32)
    # }
    ```
    If `var` is mirrored across multiple devices, then this method implements
    logic as following:
    ```python
    results = {}
    for device, v in var:
      with tf.device(device):
        # args and kwargs will be unwrapped if they are mirrored.
        results[device] = fn(v, *args, **kwargs)
    return merged(results)
    ```
    Otherwise, this method returns `fn(var, *args, **kwargs)` colocated with
    `var`.
    Args:
      var: Variable, possibly mirrored to multiple devices, to operate on.
      fn: Function to call. Should take the variable as the first argument.
      args: Tuple or list. Additional positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.
    Returns:
      By default, the merged return value of `fn` across all replicas.  The
      merged result has dependencies to make sure that if it is evaluated at
      all, the side effects (updates) will happen on every replica. If instead
      "group=False" is specified, this function will return a nest of lists
      where each list has an element per replica, and the caller is responsible
      for ensuring all elements are executed.
    """
    # TODO(b/178944108): Update the documentation to relfect the fact that
    # `update` can be called in a replica context.
    if kwargs is None:
      kwargs = {}
    replica_context = get_replica_context()
    # pylint: disable=protected-access
    if (replica_context is None or replica_context is
        _get_default_replica_context()):
      fn = autograph.tf_convert(
          fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
      with self._container_strategy().scope():
        return self._update(var, fn, args, kwargs, group)
    else:
      return self._replica_ctx_update(
          var, fn, args=args, kwargs=kwargs, group=group)
  def _update(self, var, fn, args, kwargs, group):
    raise NotImplementedError("must be implemented in descendants")
  def _local_results(self, val):
    """Returns local results per replica as a tuple."""
    if isinstance(val, ds_types.DistributedValues):
      return val._values  # pylint: disable=protected-access
    if nest.is_nested(val):
      replica_values = []
      def get_values(x, index):
        if isinstance(x, ds_types.DistributedValues):
          return x._values[index]  # pylint: disable=protected-access
        return x
      for i in range(len(self.worker_devices)):
        replica_values.append(
            nest.map_structure(
                lambda x: get_values(x, i),  # pylint: disable=cell-var-from-loop
                val))
      return tuple(replica_values)
    return (val,)
  def value_container(self, value):
    """Returns the container that this per-replica `value` belongs to.
    Args:
      value: A value returned by `run()` or a variable created in `scope()`.
    Returns:
      A container that `value` belongs to.
      If value does not belong to any container (including the case of
      container having been destroyed), returns the value itself.
      `value in experimental_local_results(value_container(value))` will
      always be true.
    """
    raise NotImplementedError("must be implemented in descendants")
  def _group(self, value, name=None):
    """Implementation of `group`."""
    value = nest.flatten(self._local_results(value))
    if len(value) != 1 or name is not None:
      return control_flow_ops.group(value, name=name)
    # Special handling for the common case of one op.
    v, = value
    if hasattr(v, "op"):
      v = v.op
    return v
  @property
  def experimental_require_static_shapes(self):
    """Returns `True` if static shape is required; `False` otherwise."""
    return self._require_static_shapes
  @property
  def _num_replicas_in_sync(self):
    """Returns number of replicas over which gradients are aggregated."""
    raise NotImplementedError("must be implemented in descendants")
  @property
  def worker_devices(self):
    """Returns the tuple of all devices used to for compute replica execution.
    """
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")
  @property
  def parameter_devices(self):
    """Returns the tuple of all devices used to place variables."""
    # TODO(josh11b): More docstring
    raise NotImplementedError("must be implemented in descendants")
  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the strategy class."""
    del session_config, cluster_spec, task_type, task_id
  def _update_config_proto(self, config_proto):
    return copy.deepcopy(config_proto)
  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings.
    Multi-worker training refers to the setup where the training is
    distributed across multiple workers, as opposed to the case where
    only a local process performs the training. This function is
    used by higher-level APIs such as Keras' `model.fit()` to infer
    for example whether or not a distribute coordinator should be run,
    and thus TensorFlow servers should be started for communication
    with other servers in the cluster, or whether or not saving/restoring
    checkpoints is relevant for preemption fault tolerance.
    Subclasses should override this to provide whether the strategy is
    currently in multi-worker setup.
    Experimental. Signature and implementation are subject to change.
    """
    raise NotImplementedError("must be implemented in descendants")
