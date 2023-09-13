@tf_export("parallel_stack")
@dispatch.add_dispatch_support
def parallel_stack(values, name="parallel_stack"):
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.
  Requires that the shape of inputs be known at graph construction time.
  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the first dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`; the `output`
  tensor will have the shape `(N, A, B, C)`.
  For example:
  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.parallel_stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
  ```
  The difference between `stack` and `parallel_stack` is that `stack` requires
  all the inputs be computed before the operation will begin but doesn't require
  that the input shapes be known during graph construction.
  `parallel_stack` will copy pieces of the input into the output as they become
  available, in some situations this can provide a performance benefit.
  Unlike `stack`, `parallel_stack` does NOT support backpropagation.
  This is the opposite of unstack.  The numpy equivalent is
      tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])
  @compatibility(eager)
  parallel_stack is not compatible with eager execution.
  @end_compatibility
  Args:
    values: A list of `Tensor` objects with the same shape and type.
    name: A name for this operation (optional).
  Returns:
    output: A stacked `Tensor` with the same type as `values`.
  Raises:
    RuntimeError: if executed in eager mode.
  """
  if context.executing_eagerly():
    raise RuntimeError("tf.parallel_stack() is not compatible with "
                       "eager execution.")
  with ops.name_scope(name):
    value_t = ops.convert_to_tensor(values[0])
    value_shape = ops.convert_to_tensor(value_t).get_shape()
    output_shape = tensor_shape.TensorShape([len(values)])
    output_shape = output_shape.concatenate(value_shape)
    # expand_dims converts concat to stack.
    return gen_array_ops.parallel_concat(
        [expand_dims(value, 0) for value in values], shape=output_shape)
