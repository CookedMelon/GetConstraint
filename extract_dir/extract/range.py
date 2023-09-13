@tf_export("range")
@dispatch.add_dispatch_support
def range(start, limit=None, delta=1, dtype=None, name="range"):  # pylint: disable=redefined-builtin
  """Creates a sequence of numbers.
  Creates a sequence of numbers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.
  The dtype of the resulting tensor is inferred from the inputs unless
  it is provided explicitly.
  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.
  For example:
  >>> start = 3
  >>> limit = 18
  >>> delta = 3
  >>> tf.range(start, limit, delta)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([ 3,  6,  9, 12, 15], dtype=int32)>
  >>> start = 3
  >>> limit = 1
  >>> delta = -0.5
  >>> tf.range(start, limit, delta)
  <tf.Tensor: shape=(4,), dtype=float32,
  numpy=array([3. , 2.5, 2. , 1.5], dtype=float32)>
  >>> limit = 5
  >>> tf.range(limit)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([0, 1, 2, 3, 4], dtype=int32)>
  Args:
    start: A 0-D `Tensor` (scalar). Acts as first entry in the range if `limit`
      is not None; otherwise, acts as range limit and first entry defaults to 0.
    limit: A 0-D `Tensor` (scalar). Upper limit of sequence, exclusive. If None,
      defaults to the value of `start` while the first entry of the range
      defaults to 0.
    delta: A 0-D `Tensor` (scalar). Number that increments `start`. Defaults to
      1.
    dtype: The type of the elements of the resulting tensor.
    name: A name for the operation. Defaults to "range".
  Returns:
    An 1-D `Tensor` of type `dtype`.
  @compatibility(numpy)
  Equivalent to np.arange
  @end_compatibility
  """
  if limit is None:
    start, limit = 0, start
  with ops.name_scope(name, "Range", [start, limit, delta]) as name:
    if not isinstance(start, ops.Tensor):
      start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    if not isinstance(limit, ops.Tensor):
      limit = ops.convert_to_tensor(limit, dtype=dtype, name="limit")
    if not isinstance(delta, ops.Tensor):
      delta = ops.convert_to_tensor(delta, dtype=dtype, name="delta")
    # infer dtype if not explicitly provided
    if dtype is None:
      dtype_hierarchy = [
          dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
      ]
      assert all(arg.dtype in dtype_hierarchy for arg in [start, limit, delta])
      inferred_dtype = max([arg.dtype for arg in [start, limit, delta]],
                           key=dtype_hierarchy.index)
    else:
      inferred_dtype = dtype
    # Always try to perform a cast even when start/limit/delta are already
    # tensors. This will resolve the case where start/limit/delta's original's
    # dtype is different from provided dtype.
    start = cast(start, inferred_dtype)
    limit = cast(limit, inferred_dtype)
    delta = cast(delta, inferred_dtype)
    return gen_math_ops._range(start, limit, delta, name=name)
