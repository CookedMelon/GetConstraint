@tf_export("sparse.eye")
def sparse_eye(num_rows,
               num_columns=None,
               dtype=dtypes.float32,
               name=None):
  """Creates a two-dimensional sparse tensor with ones along the diagonal.
  Args:
    num_rows: Non-negative integer or `int32` scalar `tensor` giving the number
      of rows in the resulting matrix.
    num_columns: Optional non-negative integer or `int32` scalar `tensor` giving
      the number of columns in the resulting matrix. Defaults to `num_rows`.
    dtype: The type of element in the resulting `Tensor`.
    name: A name for this `Op`. Defaults to "eye".
  Returns:
    A `SparseTensor` of shape [num_rows, num_columns] with ones along the
    diagonal.
  """
  with ops.name_scope(name, default_name="eye", values=[num_rows, num_columns]):
    num_rows = _make_int64_tensor(num_rows, "num_rows")
    num_columns = num_rows if num_columns is None else _make_int64_tensor(
        num_columns, "num_columns")
    # Create the sparse tensor.
    diag_size = math_ops.minimum(num_rows, num_columns)
    diag_range = math_ops.range(diag_size, dtype=dtypes.int64)
    return sparse_tensor.SparseTensor(
        indices=array_ops_stack.stack([diag_range, diag_range], axis=1),
        values=array_ops.ones(diag_size, dtype=dtype),
        dense_shape=[num_rows, num_columns])
