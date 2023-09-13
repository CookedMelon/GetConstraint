@tf_export("sparse.concat", v1=[])
def sparse_concat_v2(axis, sp_inputs, expand_nonconcat_dims=False, name=None):  # pylint: disable=missing-docstring
  sp_inputs = _convert_to_sparse_tensors(sp_inputs)
  if len(sp_inputs) == 1:  # Degenerate case of one tensor.
    return sp_inputs[0]
  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.dense_shape for sp_input in sp_inputs]
  if expand_nonconcat_dims:
    max_shape = math_ops.reduce_max(
        array_ops.concat(
            [array_ops.reshape(shape, [1, -1]) for shape in shapes], 0), 0)
    shapes = [
        array_ops.concat([
            max_shape[:axis], shape[-1:]
            if axis == -1 else shape[axis:axis + 1], []
            if axis == -1 else max_shape[axis + 1:]
        ], 0) for shape in shapes
    ]
  output_ind, output_val, output_shape = (
      gen_sparse_ops.sparse_concat(inds, vals, shapes, axis, name=name))
  input_shapes = [inp.shape for inp in sp_inputs]
  if all(shape.rank is not None for shape in input_shapes):
    if expand_nonconcat_dims:
      static_output_shape = []
      for dim in range(input_shapes[0].rank):
        static_output_shape.append(
            max(tensor_shape.dimension_at_index(shape, dim)
                for shape in input_shapes))
    else:
      static_output_shape = input_shapes[0].as_list()
    static_output_shape[axis] = sum(
        tensor_shape.dimension_at_index(shape, axis)
        for shape in input_shapes)
  else:
    static_output_shape = tensor_shape.unknown_shape()
  if all(shape.is_fully_defined() for shape in input_shapes):
    output_shape = ops.convert_to_tensor(static_output_shape,
                                         dtype=dtypes.int64)
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
  else:
    # In case there are partially defined shape, we couldn't update the
    # output_shape tensor value. We update the output._dense_shape_default,
    # which populate output.shape as the best effort.
    output = sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
    output.set_shape(tensor_shape.TensorShape(static_output_shape))
    return output
