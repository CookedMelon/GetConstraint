@tf_export("sparse.slice", v1=["sparse.slice", "sparse_slice"])
@deprecation.deprecated_endpoints("sparse_slice")
def sparse_slice(sp_input, start, size, name=None):
  """Slice a `SparseTensor` based on the `start` and `size`.
  For example, if the input is
      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]
  Graphically the output tensors are:
      sparse.slice([0, 0], [2, 4]) = shape = [2, 4]
      [    a  ]
      [b c    ]
      sparse.slice([0, 4], [2, 3]) = shape = [2, 3]
      [ d e  ]
      [      ]
  Args:
    sp_input: The `SparseTensor` to split.
    start: 1-D. tensor represents the start of the slice.
    size: 1-D. tensor represents the size of the slice.
    name: A name for the operation (optional).
  Returns:
    A `SparseTensor` objects resulting from splicing.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  start = ops.convert_to_tensor(start, dtypes.int64)
  size = ops.convert_to_tensor(size, dtypes.int64)
  with ops.name_scope(name, "SparseSlice", [sp_input]) as name:
    output_indices, output_values, output_shape = gen_sparse_ops.sparse_slice(
        sp_input.indices,
        sp_input.values,
        sp_input.dense_shape,
        start,
        size,
        name=name)
    return sparse_tensor.SparseTensor(output_indices, output_values,
                                      output_shape)
