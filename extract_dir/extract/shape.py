@tf_export("shape", v1=[])
@dispatch.add_dispatch_support
def shape_v2(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns a tensor containing the shape of the input tensor.
  See also `tf.size`, `tf.rank`.
  `tf.shape` returns a 1-D integer tensor representing the shape of `input`.
  For a scalar input, the tensor returned has a shape of (0,) and its value is
  the empty vector (i.e. []).
  For example:
  >>> tf.shape(1.)
  <tf.Tensor: shape=(0,), dtype=int32, numpy=array([], dtype=int32)>
  >>> t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  >>> tf.shape(t)
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 2, 3], dtype=int32)>
  Note: When using symbolic tensors, such as when using the Keras API,
  tf.shape() will return the shape of the symbolic tensor.
  >>> a = tf.keras.layers.Input((None, 10))
  >>> tf.shape(a)
  <... shape=(3,) dtype=int32...>
  In these cases, using `tf.Tensor.shape` will return more informative results.
  >>> a.shape
  TensorShape([None, None, 10])
  (The first `None` represents the as yet unknown batch size.)
  `tf.shape` and `Tensor.shape` should be identical in eager mode.  Within
  `tf.function` or within a `compat.v1` context, not all dimensions may be
  known until execution time. Hence, when defining custom layers and models
  for graph mode, prefer the dynamic `tf.shape(x)` over the static `x.shape`.
  Args:
    input: A `Tensor` or `SparseTensor`.
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to `tf.int32`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `out_type`.
  """
  return shape(input, name, out_type)
