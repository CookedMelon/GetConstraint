@tf_export("sparse.reorder", v1=["sparse.reorder", "sparse_reorder"])
@deprecation.deprecated_endpoints("sparse_reorder")
def sparse_reorder(sp_input, name=None):
  """Reorders a `SparseTensor` into the canonical, row-major ordering.
  Note that by convention, all sparse ops preserve the canonical ordering
  along increasing dimension number. The only time ordering can be violated
  is during manual manipulation of the indices and values to add entries.
  Reordering does not affect the shape of the `SparseTensor`.
  For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:
      [0, 3]: b
      [0, 1]: a
      [3, 1]: d
      [2, 0]: c
  then the output will be a `SparseTensor` of shape `[4, 5]` and
  `indices` / `values`:
      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d
  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional)
  Returns:
    A `SparseTensor` with the same shape and non-empty values, but in
    canonical ordering.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  reordered_ind, reordered_val = (
      gen_sparse_ops.sparse_reorder(
          sp_input.indices, sp_input.values, sp_input.dense_shape, name=name))
  if sp_input.get_shape().is_fully_defined():
    dense_shape = sp_input.get_shape().as_list()
    return sparse_tensor.SparseTensor(reordered_ind, reordered_val, dense_shape)
  else:
    dense_shape = array_ops.identity(sp_input.dense_shape)
    sp_output = sparse_tensor.SparseTensor(reordered_ind, reordered_val,
                                           dense_shape)
    # propagate the static shape
    sp_output.set_shape(sp_input.shape)
    return sp_output
