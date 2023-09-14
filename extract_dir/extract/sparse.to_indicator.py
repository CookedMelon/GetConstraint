@tf_export(
    "sparse.to_indicator", v1=["sparse.to_indicator", "sparse_to_indicator"])
@deprecation.deprecated_endpoints("sparse_to_indicator")
def sparse_to_indicator(sp_input, vocab_size, name=None):
  """Converts a `SparseTensor` of ids into a dense bool indicator tensor.
  The last dimension of `sp_input.indices` is discarded and replaced with
  the values of `sp_input`.  If `sp_input.dense_shape = [D0, D1, ..., Dn, K]`,
  then `output.shape = [D0, D1, ..., Dn, vocab_size]`, where
      output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True
  and False elsewhere in `output`.
  For example, if `sp_input.dense_shape = [2, 3, 4]` with non-empty values:
      [0, 0, 0]: 0
      [0, 1, 0]: 10
      [1, 0, 3]: 103
      [1, 1, 1]: 150
      [1, 1, 2]: 149
      [1, 1, 3]: 150
      [1, 2, 1]: 121
  and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
  tensor with False everywhere except at positions
      (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
      (1, 2, 121).
  Note that repeats are allowed in the input SparseTensor.
  This op is useful for converting `SparseTensor`s into dense formats for
  compatibility with ops that expect dense tensors.
  The input `SparseTensor` must be in row-major order.
  Args:
    sp_input: A `SparseTensor` with `values` property of type `int32` or
      `int64`.
    vocab_size: A scalar int64 Tensor (or Python int) containing the new size
      of the last dimension, `all(0 <= sp_input.values < vocab_size)`.
    name: A name prefix for the returned tensors (optional)
  Returns:
    A dense bool indicator tensor representing the indices with specified value.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  with ops.name_scope(name, "SparseToIndicator", [sp_input]) as name:
    num_entries = array_ops.shape(sp_input.indices)[0]
    new_values = array_ops.fill(array_ops.expand_dims(num_entries, 0), True)
    sp_values = sparse_tensor.SparseTensor(sp_input.indices, new_values,
                                           sp_input.dense_shape)
    sp_new = sparse_merge_impl(sp_input, sp_values, vocab_size, name)
    # validate_indices may be False because we allow duplicates in new_indices:
    # repeated indices are allowed when creating an indicator matrix.
    return sparse_tensor_to_dense(
        sp_new, default_value=False, validate_indices=False, name=name)
@tf_export(v1=["sparse.merge", "sparse_merge"])
@deprecation.deprecated(None, "No similar op available at this time.")
def sparse_merge(sp_ids, sp_values, vocab_size, name=None,
                 already_sorted=False):
  """Combines a batch of feature ids and values into a single `SparseTensor`.
  The most common use case for this function occurs when feature ids and
  their corresponding values are stored in `Example` protos on disk.
  `parse_example` will return a batch of ids and a batch of values, and this
  function joins them into a single logical `SparseTensor` for use in
  functions such as `sparse_tensor_dense_matmul`, `sparse_to_dense`, etc.
  The `SparseTensor` returned by this function has the following properties:
    - `indices` is equivalent to `sp_ids.indices` with the last
      dimension discarded and replaced with `sp_ids.values`.
    - `values` is simply `sp_values.values`.
    - If `sp_ids.dense_shape = [D0, D1, ..., Dn, K]`, then
      `output.shape = [D0, D1, ..., Dn, vocab_size]`.
  For example, consider the following feature vectors:
  ```python
    vector1 = [-3, 0, 0, 0, 0, 0]
    vector2 = [ 0, 1, 0, 4, 1, 0]
    vector3 = [ 5, 0, 0, 9, 0, 0]
  ```
  These might be stored sparsely in the following Example protos by storing
  only the feature ids (column number if the vectors are treated as a matrix)
  of the non-zero elements and the corresponding values:
  ```python
    examples = [Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0])),
                    "values": Feature(float_list=FloatList(value=[-3]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[1, 4, 3])),
                    "values": Feature(float_list=FloatList(value=[1, 1, 4]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0, 3])),
                    "values": Feature(float_list=FloatList(value=[5, 9]))})]
  ```
  The result of calling parse_example on these examples will produce a
  dictionary with entries for "ids" and "values". Passing those two objects
  to this function along with vocab_size=6, will produce a `SparseTensor` that
  sparsely represents all three instances. Namely, the `indices` property will
  contain the coordinates of the non-zero entries in the feature matrix (the
  first dimension is the row number in the matrix, i.e., the index within the
  batch, and the second dimension is the column number, i.e., the feature id);
  `values` will contain the actual values. `shape` will be the shape of the
  original matrix, i.e., (3, 6). For our example above, the output will be
  equal to:
  ```python
    SparseTensor(indices=[[0, 0], [1, 1], [1, 3], [1, 4], [2, 0], [2, 3]],
                 values=[-3, 1, 4, 1, 5, 9],
                 dense_shape=[3, 6])
  ```
  This method generalizes to higher-dimensions by simply providing a list for
  both the sp_ids as well as the vocab_size.
  In this case the resulting `SparseTensor` has the following properties:
    - `indices` is equivalent to `sp_ids[0].indices` with the last
      dimension discarded and concatenated with
      `sp_ids[0].values, sp_ids[1].values, ...`.
    - `values` is simply `sp_values.values`.
    - If `sp_ids.dense_shape = [D0, D1, ..., Dn, K]`, then
      `output.shape = [D0, D1, ..., Dn] + vocab_size`.
  Args:
    sp_ids: A single `SparseTensor` with `values` property of type `int32`
      or `int64` or a Python list of such `SparseTensor`s or a list thereof.
    sp_values: A `SparseTensor` of any type.
    vocab_size: A scalar `int64` Tensor (or Python int) containing the new size
      of the last dimension, `all(0 <= sp_ids.values < vocab_size)`.
      Or a list thereof with `all(0 <= sp_ids[i].values < vocab_size[i])` for
      all `i`.
    name: A name prefix for the returned tensors (optional)
    already_sorted: A boolean to specify whether the per-batch values in
     `sp_values` are already sorted. If so skip sorting, False by default
     (optional).
  Returns:
    A `SparseTensor` compactly representing a batch of feature ids and values,
    useful for passing to functions that expect such a `SparseTensor`.
  Raises:
    TypeError: If `sp_values` is not a `SparseTensor`. Or if `sp_ids` is neither
      a `SparseTensor` nor a list thereof. Or if `vocab_size` is not a
      `Tensor` or a Python int and `sp_ids` is a `SparseTensor`. Or if
      `vocab_size` is not a or list thereof and `sp_ids` is a list.
    ValueError: If `sp_ids` and `vocab_size` are lists of different lengths.
  """
  return sparse_merge_impl(sp_ids, sp_values, vocab_size, name, already_sorted)
def sparse_merge_impl(sp_ids,
                      sp_values,
                      vocab_size,
                      name=None,
                      already_sorted=False):
  """Internal implementation for sparse_merge to avoid deprecation warnings."""
  if isinstance(sp_ids, sparse_tensor.SparseTensorValue) or isinstance(
      sp_ids, sparse_tensor.SparseTensor):
    sp_ids = [sp_ids]
    if not (isinstance(vocab_size, ops.Tensor) or
            isinstance(vocab_size, numbers.Integral)):
      raise TypeError("vocab_size has to be a Tensor or Python int. Found %s" %
                      type(vocab_size))
    vocab_size = [vocab_size]
  else:
    if not isinstance(sp_ids, collections_abc.Iterable):
      raise TypeError("sp_ids has to be a SparseTensor or list thereof. "
                      "Found %s" % type(sp_ids))
    if not isinstance(vocab_size, collections_abc.Iterable):
      raise TypeError("vocab_size has to be a list of Tensors or Python ints. "
                      "Found %s" % type(vocab_size))
    for dim in vocab_size:
      if not (isinstance(dim, ops.Tensor) or isinstance(dim, numbers.Integral)):
        raise TypeError(
            "vocab_size has to be a list of Tensors or Python ints. Found %s" %
            type(dim))
  if len(sp_ids) != len(vocab_size):
    raise ValueError("sp_ids and vocab_size have to have equal lengths.")
  with ops.name_scope(name, "SparseMerge", [sp_ids, sp_values]):
    sp_ids = [_convert_to_sparse_tensor(sp_ids_dim) for sp_ids_dim in sp_ids]
    sp_values = _convert_to_sparse_tensor(sp_values)
    ids = []
    for sp_ids_dim in sp_ids:
      ids_dim = sp_ids_dim.values
      if sp_ids_dim.dtype != dtypes.int64:
        ids_dim = math_ops.cast(ids_dim, dtypes.int64)
      ids += [array_ops.expand_dims(ids_dim, axis=1)]
    vocab_size = [math_ops.cast(x, dtypes.int64) for x in vocab_size]
    # Slice off the last dimension of indices, then tack on the ids
    indices_columns_to_preserve = sp_ids[0].indices[:, :-1]
    new_indices = array_ops.concat([indices_columns_to_preserve] + ids, 1)
    new_values = sp_values.values
    new_shape = array_ops.concat([sp_ids[0].dense_shape[:-1], vocab_size], 0)
    result = sparse_tensor.SparseTensor(new_indices, new_values, new_shape)
    if already_sorted:
      return result
    sorted_result = sparse_reorder(result)
    return sparse_tensor.SparseTensor(
        sorted_result.indices, sorted_result.values, new_shape)
