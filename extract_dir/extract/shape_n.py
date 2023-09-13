@tf_export("shape_n")
@dispatch.add_dispatch_support
def shape_n(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns shape of a list of tensors.
  Given a list of tensors, `tf.shape_n` is much faster than applying `tf.shape`
  to each tensor individually.
  >>> a = tf.ones([1, 2])
  >>> b = tf.ones([2, 3])
  >>> c = tf.ones([3, 4])
  >>> tf.shape_n([a, b, c])
  [<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>]
  Args:
    input: A list of at least 1 `Tensor` object with the same dtype.
    out_type: The specified output type of the operation (`int32` or `int64`).
      Defaults to `tf.int32`(optional).
    name: A name for the operation (optional).
  Returns:
    A list of `Tensor` specifying the shape of each input tensor with type of
    `out_type`.
  """
  return gen_array_ops.shape_n(input, out_type=out_type, name=name)
