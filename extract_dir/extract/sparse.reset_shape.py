@tf_export(
    "sparse.reset_shape", v1=["sparse.reset_shape", "sparse_reset_shape"])
@deprecation.deprecated_endpoints("sparse_reset_shape")
def sparse_reset_shape(sp_input, new_shape=None):
  """Resets the shape of a `SparseTensor` with indices and values unchanged.
  If `new_shape` is None, returns a copy of `sp_input` with its shape reset
  to the tight bounding box of `sp_input`. This will be a shape consisting of
  all zeros if sp_input has no values.
  If `new_shape` is provided, then it must be larger or equal in all dimensions
  compared to the shape of `sp_input`. When this condition is met, the returned
  SparseTensor will have its shape reset to `new_shape` and its indices and
  values unchanged from that of `sp_input.`
  For example:
    Consider a `sp_input` with shape [2, 3, 5]:
      [0, 0, 1]: a
      [0, 1, 0]: b
      [0, 2, 2]: c
      [1, 0, 3]: d
    - It is an error to set `new_shape` as [3, 7] since this represents a
      rank-2 tensor while `sp_input` is rank-3. This is either a ValueError
      during graph construction (if both shapes are known) or an OpError during
      run time.
    - Setting `new_shape` as [2, 3, 6] will be fine as this shape is larger or
      equal in every dimension compared to the original shape [2, 3, 5].
    - On the other hand, setting new_shape as [2, 3, 4] is also an error: The
      third dimension is smaller than the original shape [2, 3, 5] (and an
      `InvalidArgumentError` will be raised).
    - If `new_shape` is None, the returned SparseTensor will have a shape
      [2, 3, 4], which is the tight bounding box of `sp_input`.
  Args:
    sp_input: The input `SparseTensor`.
    new_shape: None or a vector representing the new shape for the returned
      `SparseTensor`.
  Returns:
    A `SparseTensor` indices and values unchanged from `sp_input`. Its shape is
      `new_shape` if that is set. Otherwise it is the tight bounding box of
       `sp_input`
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
    ValueError: If `new_shape` represents a tensor with a different rank from
      that of `sp_input` (if shapes are known when graph is constructed).
    ValueError:  If `new_shape` is determined during graph build to have
      dimension sizes that are too small.
    OpError:
      - If `new_shape` has dimension sizes that are too small.
      - If shapes are not known during graph construction time, and during run
        time it is found out that the ranks do not match.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  in_indices = array_ops.identity(sp_input.indices)
  in_values = array_ops.identity(sp_input.values)
  in_shape = array_ops.identity(sp_input.dense_shape)
  if new_shape is None:
    dim_low_bound = math_ops.reduce_max(in_indices, axis=0)
    output_shape_tensor = math_ops.maximum(
        array_ops.constant(0, dtype=dtypes.int64),
        math_ops.add(dim_low_bound, array_ops.ones_like(in_shape)))
  else:
    output_shape_tensor = ops.convert_to_tensor(new_shape)
    output_shape_tensor.get_shape().assert_has_rank(1)
    output_shape_tensor = math_ops.cast(output_shape_tensor, dtypes.int64)
    # For cases when shape is known during graph construction, this catches the
    # error before the sparse_tensor.SparseTensor catches it.
    if output_shape_tensor.get_shape().rank is not None:
      output_shape_tensor.get_shape().dims[0].assert_is_compatible_with(
          in_shape.get_shape().dims[0])
    output_shape_tensor_const = tensor_util.constant_value(output_shape_tensor)
    # For cases where all shapes are known during graph construction
    if (output_shape_tensor_const is not None and
        sp_input.get_shape().is_fully_defined()):
      in_shape_const = np.array(sp_input.get_shape().as_list())
      if not np.all(in_shape_const <= output_shape_tensor_const):
        raise ValueError(
            "Requested new_shape should have dimension sizes >= sp_input.shape."
            "  Found new_shape (%s), sp_input.shape (%s)." %
            (in_shape_const, output_shape_tensor_const))
      output_shape_tensor = output_shape_tensor_const
    else:
      # For cases where shape is not known during graph construction.
      output_shape_tensor = control_flow_ops.with_dependencies([
          check_ops.assert_equal(
              array_ops.shape(in_shape), array_ops.shape(output_shape_tensor))
      ], output_shape_tensor)
      output_shape_tensor = control_flow_ops.with_dependencies(
          [check_ops.assert_less_equal(in_shape, output_shape_tensor)],
          output_shape_tensor)
  return sparse_tensor.SparseTensor(in_indices, in_values, output_shape_tensor)
