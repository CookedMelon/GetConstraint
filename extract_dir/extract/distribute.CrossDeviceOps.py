@tf_export("distribute.CrossDeviceOps")
class CrossDeviceOps(object):
  """Base class for cross-device reduction and broadcasting algorithms.
  The main purpose of this class is to be passed to
  `tf.distribute.MirroredStrategy` in order to choose among different cross
  device communication implementations. Prefer using the methods of
  `tf.distribute.Strategy` instead of the ones of this class.
  Implementations:
  * `tf.distribute.ReductionToOneDevice`
  * `tf.distribute.NcclAllReduce`
  * `tf.distribute.HierarchicalCopyAllReduce`
  """
  def __init__(self):
    self._canonicalize_devices = True
    pass
  @property
  def _num_between_graph_workers(self):
    # Returns 1 by default, the value may be overridden by sub classes.
    return 1
  def reduce(self, reduce_op, per_replica_value, destinations, options=None):
    """Reduce `per_replica_value` to `destinations`.
    See `tf.distribute.StrategyExtended.reduce_to`. This can only be called in
    the cross-replica context.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    if options is None:
      options = collective_util.Options()
    per_replica_value = _make_tensor_into_per_replica(per_replica_value)
    validate_destinations(destinations)
    # Shortcut if `per_replica_value` only contains one value.
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)
    if options is None:
      options = collective_util.Options()
    return self.reduce_implementation(reduce_op, per_replica_value,
                                      destinations, options)
  def _gather(self, per_replica_value, destinations, axis, options=None):
    """Gather `per_replica_value` to `destinations`.
    Args:
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to gather to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is gathered
        to the devices of that variable, and this method doesn't update the
        variable.
      axis: specifies the dimension to gather along within each replica's
        tensor.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`
    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    if isinstance(per_replica_value, indexed_slices.IndexedSlices):
      raise NotImplementedError("gather/all_gather does not support "
                                "IndexedSlices")
    if options is None:
      options = collective_util.Options()
    per_replica_value = _make_tensor_into_per_replica(per_replica_value)
    validate_destinations(destinations)
    # Shortcut if `per_replica_value` only contains one value.
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)
    return self._gather_implementation(per_replica_value, destinations, axis,
                                       options)
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    """Implementation of `gather` method of `tf.distribute.CrossDeviceOps`.
    Overriding this method is useful for subclass implementers.
    Args:
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to gather to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is gathered
        to the devices of that variable, this method doesn't update the
        variable.
      axis: specifies the dimension to gather along within each replica's
        tensor.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    raise NotImplementedError(
        "_gather method must be implemented in descendants.")
  def batch_reduce(self, reduce_op, value_destination_pairs, options=None):
    """Reduce values to destinations in batches.
    See `tf.distribute.StrategyExtended.batch_reduce_to`. This can only be
    called in the cross-replica context.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.CrossDeviceOps.reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.
    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    if options is None:
      options = collective_util.Options()
    # TODO(yuefengz): if destinations are different, split into several
    # `_batch_reduce` invocations.
    if not _validate_value_destination_pairs(value_destination_pairs):
      # If the first element of each pair is a tensor, we try to turn it into a
      # PerReplica object.
      value_destination_pairs = _normalize_value_destination_pairs(
          value_destination_pairs)
    for _, d in value_destination_pairs:
      validate_destinations(d)
    # Shortcut all PerReplica objects only contain one value.
    if self._num_between_graph_workers == 1 and _all_devices_match(
        value_destination_pairs, self._canonicalize_devices) and len(
            value_destination_pairs[0][0].values) == 1:
      return [
          distribute_utils.regroup(v.values, wrap_class=value_lib.Mirrored)
          for v, _ in value_destination_pairs
      ]
    if options is None:
      options = collective_util.Options()
    return self.batch_reduce_implementation(reduce_op, value_destination_pairs,
                                            options)
  def broadcast(self, tensor, destinations):
    """Broadcast `tensor` to `destinations`.
    This can only be called in the cross-replica context.
    Args:
      tensor: a `tf.Tensor` like object. The value to broadcast.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to broadcast to. Note that if it's a `tf.Variable`, the value is
        broadcasted to the devices of that variable, this method doesn't update
        the variable.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    """
    validate_destinations(destinations)
    return self.broadcast_implementation(tensor, destinations)
  @doc_controls.for_subclass_implementers
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    """Implementation of `reduce`.
    Overriding this method is useful for subclass implementers.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    raise NotImplementedError(
        "_reduce method must be implemented in descendants.")
  @doc_controls.for_subclass_implementers
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    """Implementation of `batch_reduce`.
    Overriding this method is useful for subclass implementers.
    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.
    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    raise NotImplementedError(
        "batch_reduce_implementation method must be implemented in descendants."
    )
  @doc_controls.for_subclass_implementers
  def broadcast_implementation(self, tensor, destinations):
    """Implementation of `broadcast`.
    Args:
      tensor: a `tf.Tensor` like object. The value to broadcast.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to broadcast to.
        `destinations`. Note that if it's a `tf.Variable`, the value is
        broadcasted to the devices of that variable, this method doesn't update
        the variable.
    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    """
    return simple_broadcast(
        tensor,
        destinations,
        always_mirrored=True,
        canonicalize_devices=self._canonicalize_devices)
  # ========================== Collective APIs ================================
  #
  # Different than `reduce`, `batch_reduce` and `broadcast` which must be called
  # in cross-replcia context, collective APIs are to be called in replica
  # context.
  def _all_reduce(self, reduce_op, value, replica_id, options):
    """All-reduce the `value` across all replicas so that all get the result.
    `value` can be a nested structure of tensors or `IndexedSlices`. The
    implementation should generally batch the all-reduces when possible.
    `options` can be set to hint the batching behavior.
    This API must be called in a replica context.
    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: Value to be reduced. A tensor or a nested structure of tensors or
        `IndexedSlices`.
      replica_id: An interger indicating the id of the replica where this
        all_reduce is called under. This is the local replica id that ranges
        from 0 to len(local_devices) - 1.
      options: A `tf.distribute.experimental.CommunicationOptions`.
    Returns:
      A tensor/IndexedSlices or a nested strucutre of tensors/IndexedSlices with
      the reduced values. The structure is the same as `value`.
    """
    raise NotImplementedError("_all_reduce must be implemented in descendants.")
