@tf_export("TensorArray")
class TensorArray:
  """Class wrapping dynamic-sized, per-time-step, Tensor arrays.
  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  "flow" control flow dependencies.
  Note that although the array can be read multiple times and positions can be
  overwritten, behavior may be undefined when storing multiple references to
  the same array and clear_after_read is False. In particular, avoid using
  methods like concat() to convert an intermediate TensorArray to a Tensor,
  then further modifying the TensorArray, particularly if you need to backprop
  through it later.
  Example 1: Plain reading and writing.
  >>> ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
  >>> ta = ta.write(0, 10)
  >>> ta = ta.write(1, 20)
  >>> ta = ta.write(2, 30)
  >>>
  >>> ta.read(0)
  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>
  >>> ta.read(1)
  <tf.Tensor: shape=(), dtype=float32, numpy=20.0>
  >>> ta.read(2)
  <tf.Tensor: shape=(), dtype=float32, numpy=30.0>
  >>> ta.stack()
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([10., 20., 30.],
  dtype=float32)>
  Example 2: Fibonacci sequence algorithm that writes in a loop then returns.
  >>> @tf.function
  ... def fibonacci(n):
  ...   ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  ...   ta = ta.unstack([0., 1.])
  ...
  ...   for i in range(2, n):
  ...     ta = ta.write(i, ta.read(i - 1) + ta.read(i - 2))
  ...
  ...   return ta.stack()
  >>>
  >>> fibonacci(7)
  <tf.Tensor: shape=(7,), dtype=float32,
  numpy=array([0., 1., 1., 2., 3., 5., 8.], dtype=float32)>
  Example 3: A simple loop interacting with a `tf.Variable`.
  >>> v = tf.Variable(1)
  >>> @tf.function
  ... def f(x):
  ...   ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  ...   for i in tf.range(x):
  ...     v.assign_add(i)
  ...     ta = ta.write(i, v)
  ...   return ta.stack()
  >>> f(5)
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 1,  2,  4,  7, 11],
  dtype=int32)>
  """
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    """Construct a new TensorArray or wrap an existing TensorArray handle.
    A note about the parameter `name`:
    The name of the `TensorArray` (even if passed in) is uniquified: each time
    a new `TensorArray` is created at runtime it is assigned its own name for
    the duration of the run.  This avoids name collisions if a `TensorArray`
    is created within a `while_loop`.
    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the device
        context available during its initialization.
      name: A name for the operation (optional).
    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if (context.executing_eagerly() and
        (flow is None or flow.dtype != dtypes.variant)):
      # It is possible to create a Variant-style TensorArray even in eager mode,
      # and this is fine but can have performance implications in eager.
      # An example of when this happens is if a tf.function returns a
      # TensorArray in its output; its flow variant object is returned to Eager.
      # This can be wrapped back up in a Variant-style TensorArray.
      implementation = _EagerTensorArray
    elif (flow is not None and flow.dtype == dtypes.variant or
          control_flow_util.EnableControlFlowV2(ops.get_default_graph())):
      implementation = _GraphTensorArrayV2
    else:
      implementation = _GraphTensorArray
    self._implementation = implementation(
        dtype,
        size=size,
        dynamic_size=dynamic_size,
        clear_after_read=clear_after_read,
        tensor_array_name=tensor_array_name,
        handle=handle,
        flow=flow,
        infer_shape=infer_shape,
        element_shape=element_shape,
        colocate_with_first_write_call=colocate_with_first_write_call,
        name=name)
    self._implementation.parent = weakref.ref(self)
  @property
  def flow(self):
    """The flow `Tensor` forcing ops leading to this TensorArray state."""
    return self._implementation._flow
  @property
  def dtype(self):
    """The data type of this TensorArray."""
    return self._implementation._dtype
  @property
  def handle(self):
    """The reference to the TensorArray."""
    return self._implementation.handle
  @property
  def element_shape(self):
    """The `tf.TensorShape` of elements in this TensorArray."""
    return self._implementation.element_shape
  @property
  def dynamic_size(self):
    """Python bool; if `True` the TensorArray can grow dynamically."""
    return self._implementation._dynamic_size
  @property
  def _infer_shape(self):
    # TODO(slebedev): consider making public or changing TensorArrayStructure
    # to access _implementation directly. Note that dynamic_size is also
    # only used by TensorArrayStructure.
    return self._implementation._infer_shape
  def identity(self):
    """Returns a TensorArray with the same content and properties.
    Returns:
      A new TensorArray object with flow that ensures the control dependencies
      from the contexts will become control dependencies for writes, reads, etc.
      Use this object for all subsequent operations.
    """
    return self._implementation.identity()
  def grad(self, source, flow=None, name=None):
    return self._implementation.grad(source, flow=flow, name=name)
  def read(self, index, name=None):
    """Read the value at location `index` in the TensorArray.
    Args:
      index: 0-D.  int32 tensor with the index to read from.
      name: A name for the operation (optional).
    Returns:
      The tensor at index `index`.
    """
    return self._implementation.read(index, name=name)
  @tf_should_use.should_use_result(warn_in_eager=True)
  def write(self, index, value, name=None):
    """Write `value` into index `index` of the TensorArray.
    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The Tensor to write to this index.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the write occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if there are more writers than specified.
    """
    return self._implementation.write(index, value, name=name)
  def stack(self, name=None):
    """Return the values in the TensorArray as a stacked `Tensor`.
    All of the values must have been written and their shapes must all match.
    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.
    For example:
    >>> ta = tf.TensorArray(tf.int32, size=3)
    >>> ta = ta.write(0, tf.constant([1, 2]))
    >>> ta = ta.write(1, tf.constant([3, 4]))
    >>> ta = ta.write(2, tf.constant([5, 6]))
    >>> ta.stack()
    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)>
    Args:
      name: A name for the operation (optional).
    Returns:
      All the tensors in the TensorArray stacked into one tensor.
    """
    return self._implementation.stack(name=name)
  def gather(self, indices, name=None):
    """Return selected values in the TensorArray as a packed `Tensor`.
    All of selected values must have been written and their shapes
    must all match.
    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the
        `TensorArray` is not dynamic, `max_value=size()`.
      name: A name for the operation (optional).
    Returns:
      The tensors in the `TensorArray` selected by `indices`, packed into one
      tensor.
    """
    return self._implementation.gather(indices, name=name)
  def concat(self, name=None):
    """Return the values in the TensorArray as a concatenated `Tensor`.
    All of the values must have been written, their ranks must match, and
    and their shapes must all match for all dimensions except the first.
    Args:
      name: A name for the operation (optional).
    Returns:
      All the tensors in the TensorArray concatenated into one tensor.
    """
    return self._implementation.concat(name=name)
  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    """Unstack the values of a `Tensor` in the TensorArray.
    If input value shapes have rank-`R`, then the output TensorArray will
    contain elements whose shapes are rank-`(R-1)`.
    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unstack.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the unstack occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.unstack(value, name=name)
  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    """Scatter the values of a `Tensor` in specific indices of a `TensorArray`.
    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the
        `TensorArray` is not dynamic, `max_value=size()`.
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the scatter occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.scatter(indices, value, name=name)
  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    """Split the values of a `Tensor` into the TensorArray.
    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to split.
      lengths: 1-D.  int32 vector with the lengths to use when splitting `value`
        along its first dimension.
      name: A name for the operation (optional).
    Returns:
      A new TensorArray object with flow that ensures the split occurs.
      Use this object for all subsequent operations.
    Raises:
      ValueError: if the shape inference fails.
    """
    return self._implementation.split(value, lengths, name=name)
  def size(self, name=None):
    """Return the size of the TensorArray."""
    return self._implementation.size(name=name)
  @tf_should_use.should_use_result
  def close(self, name=None):
    """Close the current TensorArray."""
    return self._implementation.close(name=name)
