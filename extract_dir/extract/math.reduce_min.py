@tf_export("math.reduce_min", "reduce_min", v1=[])
@dispatch.add_dispatch_support
def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the `tf.math.minimum` of elements across dimensions of a tensor.
  This is the reduction operation for the elementwise `tf.math.minimum` op.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  For example:
  >>> a = tf.constant([
  ...   [[1, 2], [3, 4]],
  ...   [[1, 2], [3, 4]]
  ... ])
  >>> tf.reduce_min(a)
  <tf.Tensor: shape=(), dtype=int32, numpy=1>
  Choosing a specific axis returns minimum element in the given axis:
  >>> b = tf.constant([[1, 2, 3], [4, 5, 6]])
  >>> tf.reduce_min(b, axis=0)
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
  >>> tf.reduce_min(b, axis=1)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 4], dtype=int32)>
  Setting `keepdims` to `True` retains the dimension of `input_tensor`:
  >>> tf.reduce_min(a, keepdims=True)
  <tf.Tensor: shape=(1, 1, 1), dtype=int32, numpy=array([[[1]]], dtype=int32)>
  >>> tf.math.reduce_min(a, axis=0, keepdims=True)
  <tf.Tensor: shape=(1, 2, 2), dtype=int32, numpy=
  array([[[1, 2],
          [3, 4]]], dtype=int32)>
  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor.
  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility
  """
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._min(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))
