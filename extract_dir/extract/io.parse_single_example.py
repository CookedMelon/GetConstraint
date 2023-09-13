@tf_export("io.parse_single_example", v1=[])
@dispatch.add_dispatch_support
def parse_single_example_v2(
    serialized, features, example_names=None, name=None
    ):
  """Parses a single `Example` proto.
  Similar to `parse_example`, except:
  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.
  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).
  One might see performance advantages by batching `Example` protos with
  `parse_example` instead of using this function directly.
  Args:
    serialized: A scalar string Tensor, a single serialized Example.
    features: A mapping of feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    example_names: (Optional) A scalar string Tensor, the associated name.
    name: A name for this operation (optional).
  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.
  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Invalid argument: features cannot be None.")
  with ops.name_scope(name, "ParseSingleExample", [serialized, example_names]):
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    serialized = _assert_scalar(serialized, "serialized")
    return parse_example_v2(serialized, features, example_names, name)
