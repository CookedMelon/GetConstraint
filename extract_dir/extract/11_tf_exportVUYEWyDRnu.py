"/home/cc/Workspace/tfconstraint/python/ops/sparse_ops.py"
@tf_export(
    "sparse.fill_empty_rows",
    v1=["sparse.fill_empty_rows", "sparse_fill_empty_rows"])
@deprecation.deprecated_endpoints("sparse_fill_empty_rows")
def sparse_fill_empty_rows(sp_input, default_value, name=None):
  """Fills empty rows in the input 2-D `SparseTensor` with a default value.
  This op adds entries with the specified `default_value` at index
  `[row, 0]` for any row in the input that does not already have a value.
  For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d
  Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
      [0, 1]: a
      [0, 3]: b
      [1, 0]: default_value
      [2, 0]: c
      [3, 1]: d
      [4, 0]: default_value
  Note that the input may have empty columns at the end, with no effect on
  this op.
  The output `SparseTensor` will be in row-major order and will have the
  same shape as the input.
  This op also returns an indicator vector such that
      empty_row_indicator[i] = True iff row i was an empty row.
  Args:
    sp_input: A `SparseTensor` with shape `[N, M]`.
    default_value: The value to fill for empty rows, with the same type as
      `sp_input.`
    name: A name prefix for the returned tensors (optional)
  Returns:
    sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
      rows filled in with `default_value`.
    empty_row_indicator: A bool vector of length `N` indicating whether each
      input row was empty.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  with ops.name_scope(name, "SparseFillEmptyRows", [sp_input]):
    default_value = ops.convert_to_tensor(
        default_value, dtype=sp_input.values.dtype)
    (output_indices, output_values, empty_row_indicator,
     unused_reverse_index_map) = gen_sparse_ops.sparse_fill_empty_rows(
         indices=sp_input.indices,
         values=sp_input.values,
         dense_shape=sp_input.dense_shape,
         default_value=default_value)
    return (sparse_tensor.SparseTensor(
        indices=output_indices,
        values=output_values,
        dense_shape=sp_input.dense_shape), empty_row_indicator)
@tf_export(v1=["io.serialize_sparse", "serialize_sparse"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("serialize_sparse")
def serialize_sparse(sp_input, name=None, out_type=dtypes.string):
  """Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.
  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.
  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  return serialize_sparse_v2(sp_input, out_type, name)
