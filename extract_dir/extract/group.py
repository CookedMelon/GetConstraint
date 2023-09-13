@tf_export("group")
def group(*inputs, **kwargs):
  """Create an op that groups multiple operations.
  When this op finishes, all ops in `inputs` have finished. This op has no
  output.
  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.group` when working with v1
  `tf.Graph` code.
  When operating in a v1-style graph context, ops are not executed in the same
  order as specified in the code; TensorFlow will attempt to execute ops in
  parallel or in an order convenient to the result it is computing.  `tf.group`
  allows you to request that one or more results finish before execution
  continues.
  `tf.group` creates a single op (of type `NoOp`), and then adds appropriate
  control dependencies.  Thus, `c = tf.group(a, b)` will compute the same graph
  as this:
      with tf.control_dependencies([a, b]):
          c = tf.no_op()
  See also `tf.tuple` and
  `tf.control_dependencies`.
  Args:
    *inputs: Zero or more tensors to group.
    name: A name for this operation (optional).
  Returns:
    An Operation that executes all its inputs.
  Raises:
    ValueError: If an unknown keyword argument is provided.
  """
  if context.executing_eagerly():
    return None
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: " + ", ".join(kwargs.keys()))
  with ops.name_scope(name, "group_deps", inputs) as name:
    # Grouping no inputs means do nothing
    if not inputs:
      return no_op(name=name)
    # Sorts *inputs according to their devices.
    ops_on_device = {}  # device -> operations specified on the device.
    for inp in nest.flatten(inputs, expand_composites=True):
      if not hasattr(inp, "device"):
        raise TypeError("'inputs' should be zero or more (nested) Tensors. "
                        f"Received '{inp}' with type '{type(inp)}'.")
      dev = inp.device
      if dev in ops_on_device:
        ops_on_device[dev].append(inp)
      else:
        ops_on_device[dev] = [inp]
    if len(ops_on_device) == 1:
      # 1-level tree. The root node is the returned NoOp node.
      (dev, deps), = ops_on_device.items()
      return _GroupControlDeps(dev, deps, name=name)
    # 2-level tree. The root node is the returned NoOp node.
    # deps contains 1 NoOp node for each device.
    deps = []
    def device_key(dev):
      """A sort key that allows None to be compared to strings."""
      return "" if dev is None else dev
    for dev in sorted(ops_on_device, key=device_key):
      deps.append(_GroupControlDeps(dev, ops_on_device[dev]))
    with ops.control_dependencies(deps):
      return no_op(name=name)
