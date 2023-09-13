@tf_export("one_hot")
@dispatch.add_dispatch_support
def one_hot(indices,
            depth,
            on_value=None,
            off_value=None,
            axis=None,
            dtype=None,
            name=None):
  """Returns a one-hot tensor.
  See also `tf.fill`, `tf.eye`.
  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.
  `on_value` and `off_value` must have matching data types. If `dtype` is also
  provided, they must be the same data type as specified by `dtype`.
  If `on_value` is not provided, it will default to the value `1` with type
  `dtype`
  If `off_value` is not provided, it will default to the value `0` with type
  `dtype`
  If the input `indices` is rank `N`, the output will have rank `N+1`. The
  new axis is created at dimension `axis` (default: the new axis is appended
  at the end).
  If `indices` is a scalar the output shape will be a vector of length `depth`
  If `indices` is a vector of length `features`, the output shape will be:
  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```
  If `indices` is a matrix (batch) with shape `[batch, features]`, the output
  shape will be:
  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```
  If `indices` is a RaggedTensor, the 'axis' argument must be positive and refer
  to a non-ragged axis. The output will be equivalent to applying 'one_hot' on
  the values of the RaggedTensor, and creating a new RaggedTensor from the
  result.
  If `dtype` is not provided, it will attempt to assume the data type of
  `on_value` or `off_value`, if one or both are passed in. If none of
  `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
  value `tf.float32`.
  Note: If a non-numeric data type output is desired (`tf.string`, `tf.bool`,
  etc.), both `on_value` and `off_value` _must_ be provided to `one_hot`.
  For example:
  ```python
  indices = [0, 1, 2]
  depth = 3
  tf.one_hot(indices, depth)  # output: [3 x 3]
  # [[1., 0., 0.],
  #  [0., 1., 0.],
  #  [0., 0., 1.]]
  indices = [0, 2, -1, 1]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=5.0, off_value=0.0,
             axis=-1)  # output: [4 x 3]
  # [[5.0, 0.0, 0.0],  # one_hot(0)
  #  [0.0, 0.0, 5.0],  # one_hot(2)
  #  [0.0, 0.0, 0.0],  # one_hot(-1)
  #  [0.0, 5.0, 0.0]]  # one_hot(1)
  indices = [[0, 2], [1, -1]]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=1.0, off_value=0.0,
             axis=-1)  # output: [2 x 2 x 3]
  # [[[1.0, 0.0, 0.0],   # one_hot(0)
  #   [0.0, 0.0, 1.0]],  # one_hot(2)
  #  [[0.0, 1.0, 0.0],   # one_hot(1)
  #   [0.0, 0.0, 0.0]]]  # one_hot(-1)
  indices = tf.ragged.constant([[0, 1], [2]])
  depth = 3
  tf.one_hot(indices, depth)  # output: [2 x None x 3]
  # [[[1., 0., 0.],
  #   [0., 1., 0.]],
  #  [[0., 0., 1.]]]
  ```
  Args:
    indices: A `Tensor` of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.
    name: A name for the operation (optional).
  Returns:
    output: The one-hot tensor.
  Raises:
    TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
    TypeError: If dtype of `on_value` and `off_value` don't match one another
  """
  with ops.name_scope(
      name, "one_hot",
      [indices, depth, on_value, off_value, axis, dtype]) as name:
    on_exists = on_value is not None
    off_exists = off_value is not None
    if on_exists:
      on_value = ops.convert_to_tensor(on_value, dtype_hint=dtype)
    if off_exists:
      off_value = ops.convert_to_tensor(off_value, dtype_hint=dtype)
    on_dtype = on_value.dtype.base_dtype if on_exists else None
    off_dtype = off_value.dtype.base_dtype if off_exists else None
    if on_exists or off_exists:
      if dtype is not None:
        # Ensure provided on_value and/or off_value match dtype
        if on_exists and on_dtype != dtype:
          raise TypeError("dtype {0} of on_value does not match "
                          "dtype parameter {1}".format(on_dtype, dtype))
        if off_exists and off_dtype != dtype:
          raise TypeError("dtype {0} of off_value does not match "
                          "dtype parameter {1}".format(off_dtype, dtype))
      else:
        # dtype not provided: automatically assign it
        dtype = on_dtype if on_exists else off_dtype
    elif dtype is None:
      # None of on_value, off_value, or dtype provided. Default dtype to float32
      dtype = dtypes.float32
    if not on_exists:
      # on_value not provided: assign to value 1 of type dtype
      on_value = ops.convert_to_tensor(1, dtype, name="on_value")
      on_dtype = dtype
    if not off_exists:
      # off_value not provided: assign to value 0 of type dtype
      off_value = ops.convert_to_tensor(0, dtype, name="off_value")
      off_dtype = dtype
    if on_dtype != off_dtype:
      raise TypeError("dtype {0} of on_value does not match "
                      "dtype {1} of off_value".format(on_dtype, off_dtype))
    return gen_array_ops.one_hot(indices, depth, on_value, off_value, axis,
                                 name)
