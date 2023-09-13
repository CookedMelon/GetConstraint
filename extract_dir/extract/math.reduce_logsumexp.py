@tf_export("math.reduce_logsumexp", "reduce_logsumexp", v1=[])
@dispatch.add_dispatch_support
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.
  For example:
  ```python
  x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
  tf.reduce_logsumexp(x)  # log(6)
  tf.reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
  tf.reduce_logsumexp(x, 1)  # [log(3), log(3)]
  tf.reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
  tf.reduce_logsumexp(x, [0, 1])  # log(6)
  ```
  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor.
  """
  with ops.name_scope(name, "ReduceLogSumExp", [input_tensor]) as name:
    raw_max = reduce_max(input_tensor, axis=axis, keepdims=True)
    my_max = array_ops.stop_gradient(
        gen_math_ops.select(
            gen_math_ops.is_finite(raw_max), raw_max,
            gen_array_ops.zeros_like(raw_max)))
    result = gen_math_ops.log(
        reduce_sum(
            exp(subtract(input_tensor, my_max)),
            axis=axis,
            keepdims=keepdims))
    if not keepdims:
      my_max = array_ops.reshape(my_max, gen_array_ops.shape(result))
    result = add(result, my_max, name=name)
    return _may_reduce_to_scalar(keepdims, axis, result)
