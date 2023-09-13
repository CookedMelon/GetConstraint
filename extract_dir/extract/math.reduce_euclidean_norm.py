@tf_export("math.reduce_euclidean_norm")
@dispatch.add_dispatch_support
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the Euclidean norm of elements across dimensions of a tensor.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  For example:
  ```python
  x = tf.constant([[1, 2, 3], [1, 1, 1]]) # x.dtype is tf.int32
  tf.math.reduce_euclidean_norm(x)  # returns 4 as dtype is tf.int32
  y = tf.constant([[1, 2, 3], [1, 1, 1]], dtype = tf.float32)
  tf.math.reduce_euclidean_norm(y)  # returns 4.1231055 which is sqrt(17)
  tf.math.reduce_euclidean_norm(y, 0)  # [sqrt(2), sqrt(5), sqrt(10)]
  tf.math.reduce_euclidean_norm(y, 1)  # [sqrt(14), sqrt(3)]
  tf.math.reduce_euclidean_norm(y, 1, keepdims=True)  # [[sqrt(14)], [sqrt(3)]]
  tf.math.reduce_euclidean_norm(y, [0, 1])  # sqrt(17)
  ```
  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor, of the same dtype as the input_tensor.
  """
  keepdims = bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.euclidean_norm(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))
