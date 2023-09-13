@tf_export("math.reduce_sum", "reduce_sum", v1=[])
@dispatch.add_dispatch_support
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the sum of elements across dimensions of a tensor.
  This is the reduction operation for the elementwise `tf.math.add` op.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  For example:
    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
    >>> x.numpy()
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> tf.reduce_sum(x).numpy()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> tf.reduce_sum(x, 0).numpy()
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> tf.reduce_sum(x, 1).numpy()
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> tf.reduce_sum(x, 1, keepdims=True).numpy()
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> tf.reduce_sum(x, [0, 1]).numpy()
    6
  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor)]`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor, of the same dtype as the input_tensor.
  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """
  return reduce_sum_with_dims(input_tensor, axis, keepdims, name,
                              _ReductionDims(input_tensor, axis))
