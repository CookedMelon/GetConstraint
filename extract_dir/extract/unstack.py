@tf_export("unstack")
@dispatch.add_dispatch_support
def unstack(value, num=None, axis=0, name="unstack"):
  """Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
  Unpacks tensors from `value` by chipping it along the `axis` dimension.
  >>> x = tf.reshape(tf.range(12), (3,4))
  >>>
  >>> p, q, r = tf.unstack(x)
  >>> p.shape.as_list()
  [4]
  >>> i, j, k, l = tf.unstack(x, axis=1)
  >>> i.shape.as_list()
  [3]
  This is the opposite of stack.
  >>> x = tf.stack([i, j, k, l], axis=1)
  More generally if you have a tensor of shape `(A, B, C, D)`:
  >>> A, B, C, D = [2, 3, 4, 5]
  >>> t = tf.random.normal(shape=[A, B, C, D])
  The number of tensor returned is equal to the length of the target `axis`:
  >>> axis = 2
  >>> items = tf.unstack(t, axis=axis)
  >>> len(items) == t.shape[axis]
  True
  The shape of each result tensor is equal to the shape of the input tensor,
  with the target `axis` removed.
  >>> items[0].shape.as_list()  # [A, B, D]
  [2, 3, 5]
  The value of each tensor `items[i]` is equal to the slice of `input` across
  `axis` at index `i`:
  >>> for i in range(len(items)):
  ...   slice = t[:,:,i,:]
  ...   assert tf.reduce_all(slice == items[i])
  #### Python iterable unpacking
  With eager execution you _can_ unstack the 0th axis of a tensor using python's
  iterable unpacking:
  >>> t = tf.constant([1,2,3])
  >>> a,b,c = t
  `unstack` is still necessary because Iterable unpacking doesn't work in
  a `@tf.function`: Symbolic tensors are not iterable.
  You need to use `tf.unstack` here:
  >>> @tf.function
  ... def bad(t):
  ...   a,b,c = t
  ...   return a
  >>>
  >>> bad(t)
  Traceback (most recent call last):
  ...
  OperatorNotAllowedInGraphError: ...
  >>> @tf.function
  ... def good(t):
  ...   a,b,c = tf.unstack(t)
  ...   return a
  >>>
  >>> good(t).numpy()
  1
  #### Unknown shapes
  Eager tensors have concrete values, so their shape is always known.
  Inside a `tf.function` the symbolic tensors may have unknown shapes.
  If the length of `axis` is unknown `tf.unstack` will fail because it cannot
  handle an unknown number of tensors:
  >>> @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  ... def bad(t):
  ...   tensors = tf.unstack(t)
  ...   return tensors[0]
  >>>
  >>> bad(tf.constant([1.0, 2.0, 3.0]))
  Traceback (most recent call last):
  ...
  ValueError: Cannot infer argument `num` from shape (None,)
  If you know the `axis` length you can pass it as the `num` argument. But this
  must be a constant value.
  If you actually need a variable number of tensors in a single `tf.function`
  trace, you will need to use exlicit loops and a `tf.TensorArray` instead.
  Args:
    value: A rank `R > 0` `Tensor` to be unstacked.
    num: An `int`. The length of the dimension `axis`. Automatically inferred if
      `None` (the default).
    axis: An `int`. The axis to unstack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-R, R)`.
    name: A name for the operation (optional).
  Returns:
    The list of `Tensor` objects unstacked from `value`.
  Raises:
    ValueError: If `axis` is out of the range `[-R, R)`.
    ValueError: If `num` is unspecified and cannot be inferred.
    InvalidArgumentError: If `num` does not match the shape of `value`.
  """
  if num is None:
    value = ops.convert_to_tensor(value)
    value_shape = value.get_shape()
    if value_shape.ndims is not None:
      if axis < -value_shape.ndims or axis >= value_shape.ndims:
        raise ValueError(f"Argument `axis` = {axis} not in range "
                         f"[{-value_shape.ndims}, {value_shape.ndims})")
      num = value_shape.dims[axis].value
    if num is None:
      raise ValueError(f"Cannot infer argument `num` from shape {value_shape}")
  return gen_array_ops.unpack(value, num=num, axis=axis, name=name)
