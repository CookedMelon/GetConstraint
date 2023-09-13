@tf_export("unique")
@dispatch.add_dispatch_support
def unique(x, out_idx=dtypes.int32, name=None):
  """Finds unique elements in a 1-D tensor.
  See also `tf.unique_with_counts`.
  This operation returns a tensor `y` containing all the unique elements
  of `x` sorted in the same order that they occur in `x`. This operation
  also returns a tensor `idx` the same size as `x` that contains the index
  of each value of `x` in the unique output `y`. In other words:
    y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]
  Example usage:
  >>> x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
  >>> y, idx = unique(x)
  >>> y
  <tf.Tensor: id=5, shape=(5,), dtype=int32,
  numpy=array([1, 2, 4, 7, 8], dtype=int32)>
  >>> idx
  <tf.Tensor: id=6, shape=(9,), dtype=int32,
  numpy=array([0, 0, 1, 2, 2, 2, 3, 4, 4], dtype=int32)>
  Args:
    x: A Tensor. 1-D.
    out_idx: An optional tf.DType from: tf.int32, tf.int64. Defaults to
      tf.int32.
    name: A name for the operation (optional).
  Returns:
    A tuple of Tensor objects (y, idx).
      y: A Tensor. Has the same type as x.
      idx: A Tensor of type out_idx.
  """
  # TODO(yongtang): switch to v2 once API deprecation
  # period (3 weeks) pass.
  # TODO(yongtang): The documentation should also
  # be updated when switch  to v2.
  return gen_array_ops.unique(x, out_idx, name)
