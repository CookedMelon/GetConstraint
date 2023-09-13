@tf_export("rank")
@dispatch.add_dispatch_support
def rank(input, name=None):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor.
  See also `tf.shape`.
  Returns a 0-D `int32` `Tensor` representing the rank of `input`.
  For example:
  ```python
  # shape of tensor 't' is [2, 2, 3]
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.rank(t)  # 3
  ```
  **Note**: The rank of a tensor is not the same as the rank of a matrix. The
  rank of a tensor is the number of indices required to uniquely select each
  element of the tensor. Rank is also known as "order", "degree", or "ndims."
  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `int32`.
  @compatibility(numpy)
  Equivalent to np.ndim
  @end_compatibility
  """
  return rank_internal(input, name, optimize=True)
