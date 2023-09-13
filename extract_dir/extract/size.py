@tf_export("size", v1=[])
@dispatch.add_dispatch_support
def size_v2(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns the size of a tensor.
  See also `tf.shape`.
  Returns a 0-D `Tensor` representing the number of elements in `input`
  of type `out_type`. Defaults to tf.int32.
  For example:
  >>> t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  >>> tf.size(t)
  <tf.Tensor: shape=(), dtype=int32, numpy=12>
  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    out_type: (Optional) The specified non-quantized numeric output type of the
      operation. Defaults to `tf.int32`.
  Returns:
    A `Tensor` of type `out_type`. Defaults to `tf.int32`.
  @compatibility(numpy)
  Equivalent to np.size()
  @end_compatibility
  """
  return size(input, name, out_type)
