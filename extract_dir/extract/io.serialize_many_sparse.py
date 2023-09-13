@tf_export("io.serialize_many_sparse", v1=[])
@dispatch.add_dispatch_support
def serialize_many_sparse_v2(sp_input, out_type=dtypes.string, name=None):
  """Serialize `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor`.
  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.
  The minibatch size `N` is extracted from `sparse_shape[0]`.
  Args:
    sp_input: The input rank `R` `SparseTensor`.
    out_type: The `dtype` to use for serialization.
    name: A name prefix for the returned tensors (optional).
  Returns:
    A matrix (2-D `Tensor`) with `N` rows and `3` columns. Each column
    represents serialized `SparseTensor`'s indices, values, and shape
    (respectively).
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  return gen_sparse_ops.serialize_many_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      name=name,
      out_type=out_type)
