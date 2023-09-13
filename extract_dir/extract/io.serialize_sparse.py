@tf_export("io.serialize_sparse", v1=[])
@dispatch.add_dispatch_support
def serialize_sparse_v2(sp_input, out_type=dtypes.string, name=None):
  """Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.
  Args:
    sp_input: The input `SparseTensor`.
    out_type: The `dtype` to use for serialization.
    name: A name prefix for the returned tensors (optional).
  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  return gen_sparse_ops.serialize_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      name=name,
      out_type=out_type)
