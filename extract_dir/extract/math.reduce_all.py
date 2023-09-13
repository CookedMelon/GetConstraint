@tf_export("math.reduce_all", "reduce_all", v1=[])
@dispatch.add_dispatch_support
def reduce_all(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.logical_and` of elements across dimensions of a tensor.
  This is the reduction operation for the elementwise `tf.math.logical_and` op.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  For example:
    >>> x = tf.constant([[True,  True], [False, False]])
    >>> tf.math.reduce_all(x)
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    >>> tf.math.reduce_all(x, 0)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False, False])>
    >>> tf.math.reduce_all(x, 1)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>
  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor.
  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility
  """
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._all(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))
