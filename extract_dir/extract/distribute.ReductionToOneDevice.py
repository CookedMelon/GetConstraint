@tf_export("distribute.ReductionToOneDevice")
class ReductionToOneDevice(CrossDeviceOps):
  """A CrossDeviceOps implementation that copies values to one device to reduce.
  This implementation always copies values to one device to reduce them, then
  broadcast reduced values to the destinations. It doesn't support efficient
  batching.
  Here is how you can use `ReductionToOneDevice` in
  `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.ReductionToOneDevice())
  ```
  """
  def __init__(self, reduce_to_device=None, accumulation_fn=None):
    """Initializes with a device to reduce to and a way to accumulate.
    Args:
      reduce_to_device: the intermediate device to reduce to. If None, reduce
        to the first device in `destinations` of the `reduce` method.
      accumulation_fn: a function that does accumulation.  If None,
        `tf.math.add_n` is used.
    """
    self.reduce_to_device = reduce_to_device
    self.accumulation_fn = accumulation_fn or math_ops.add_n
    super(ReductionToOneDevice, self).__init__()
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    del options  # Unused.
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Reduce to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    reduced = _simple_reduce(per_replica_value, reduce_to_device,
                             self.accumulation_fn, reduce_op)
    return self.broadcast(reduced, destinations)
  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    del options  # Unused.
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Gather to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    gathered = _simple_gather(per_replica_value, reduce_to_device, axis)
    return self.broadcast(gathered, destinations)
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    return [
        self.reduce_implementation(
            reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]
