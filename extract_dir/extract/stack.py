@tf_export("stack")
@dispatch.add_dispatch_support
def stack(values, axis=0, name="stack"):
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
  See also `tf.concat`, `tf.tile`, `tf.repeat`.
  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`;
  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.
  For example:
  >>> x = tf.constant([1, 4])
  >>> y = tf.constant([2, 5])
  >>> z = tf.constant([3, 6])
  >>> tf.stack([x, y, z])
  <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
  array([[1, 4],
         [2, 5],
         [3, 6]], dtype=int32)>
  >>> tf.stack([x, y, z], axis=1)
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[1, 2, 3],
         [4, 5, 6]], dtype=int32)>
  This is the opposite of unstack.  The numpy equivalent is `np.stack`
  >>> np.array_equal(np.stack([x, y, z]), tf.stack([x, y, z]))
  True
  Args:
    values: A list of `Tensor` objects with the same shape and type.
    axis: An `int`. The axis to stack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
    name: A name for this operation (optional).
  Returns:
    output: A stacked `Tensor` with the same type as `values`.
  Raises:
    ValueError: If `axis` is out of the range [-(R+1), R+1).
  """
  if axis == 0:
    try:
      # If the input is a constant list, it can be converted to a constant op
      return ops.convert_to_tensor(values, name=name)
    except (TypeError, ValueError, NotImplementedError):
      pass  # Input list contains non-constant tensors
  value_shape = ops.convert_to_tensor(values[0], name=name)._shape_tuple()  # pylint: disable=protected-access
  if value_shape is not None:
    expanded_num_dims = len(value_shape) + 1
    if axis < -expanded_num_dims or axis >= expanded_num_dims:
      raise ValueError(f"Argument `axis` = {axis} not in range "
                       f"[{-expanded_num_dims}, {expanded_num_dims})")
  return gen_array_ops.pack(values, axis=axis, name=name)
