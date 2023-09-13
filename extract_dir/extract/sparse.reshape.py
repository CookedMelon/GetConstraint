@tf_export("sparse.reshape", v1=["sparse.reshape", "sparse_reshape"])
@deprecation.deprecated_endpoints("sparse_reshape")
@dispatch.add_dispatch_support
def sparse_reshape(sp_input, shape, name=None):
  """Reshapes a `SparseTensor` to represent values in a new dense shape.
  This operation has the same semantics as `reshape` on the represented dense
  tensor.  The indices of non-empty values in `sp_input` are recomputed based
  on the new dense shape, and a new `SparseTensor` is returned containing the
  new indices and new shape.  The order of non-empty values in `sp_input` is
  unchanged.
  If one component of `shape` is the special value -1, the size of that
  dimension is computed so that the total dense size remains constant.  At
  most one component of `shape` can be -1.  The number of dense elements
  implied by `shape` must be the same as the number of dense elements
  originally represented by `sp_input`.
  For example, if `sp_input` has shape `[2, 3, 6]` and `indices` / `values`:
      [0, 0, 0]: a
      [0, 0, 1]: b
      [0, 1, 0]: c
      [1, 0, 0]: d
      [1, 2, 3]: e
  and `shape` is `[9, -1]`, then the output will be a `SparseTensor` of
  shape `[9, 4]` and `indices` / `values`:
      [0, 0]: a
      [0, 1]: b
      [1, 2]: c
      [4, 2]: d
      [8, 1]: e
  Args:
    sp_input: The input `SparseTensor`.
    shape: A 1-D (vector) int64 `Tensor` specifying the new dense shape of the
      represented `SparseTensor`.
    name: A name prefix for the returned tensors (optional)
  Returns:
    A `SparseTensor` with the same non-empty values but with indices calculated
    by the new dense shape.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
    ValueError:  If argument `shape` requests a `SparseTensor` with a different
      number of elements than `sp_input`.
    ValueError:  If `shape` has more than one inferred (== -1) dimension.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  shape = math_ops.cast(shape, dtype=dtypes.int64)
  with ops.name_scope(name, "SparseReshape", [sp_input]) as name:
    reshaped_ind, reshaped_shape = gen_sparse_ops.sparse_reshape(
        sp_input.indices, sp_input.dense_shape, shape, name=name)
    reshaped_shape_const = tensor_util.constant_value_as_shape(shape)
    reshaped_shape_const = (
        reshaped_shape_const.as_list() if reshaped_shape_const.ndims is not None
        else None)
    if (reshaped_shape_const is not None
        and sp_input.shape.is_fully_defined()):
      # constant_value_as_shape tends to get more information about the partial
      # shape values, but here we specifically need to know if the *user* passed
      # a shape with 2+ unknown dimensions; and for that constant_value
      # provides either the user's direct value or None if only partial elements
      # are known via the python shape inference code.
      shape_const_by_user = tensor_util.constant_value(shape)
      if shape_const_by_user is not None:
        num_implied_by_user = sum(d == -1 for d in shape_const_by_user)
        if num_implied_by_user > 1:
          raise ValueError(
              "At most one dimension can be inferred (-1). Found: %s"
              % shape_const_by_user)
      original_reshaped_shape = list(reshaped_shape_const)  # A copy
      in_shape_size = np.prod(sp_input.shape.as_list())
      num_implied = sum(dim is None for dim in reshaped_shape_const)
      # If there is a 0 dim in the user-provided shape, we cannot infer the
      # unknown dim reliably. This is why we skip the `if` branch below when
      # a 0 is present in `reshaped_shape_const`. Same below.
      if num_implied == 1 and 0 not in reshaped_shape_const:
        implied_idx = original_reshaped_shape.index(None)
        non_implied_idx = (
            original_reshaped_shape[:implied_idx] +
            original_reshaped_shape[implied_idx + 1:])
        reshaped_shape_const[implied_idx] = int(
            in_shape_size // np.prod(non_implied_idx))
      if num_implied == 0 or (num_implied == 1 and
                              0 not in reshaped_shape_const):
        reshaped_size = np.prod(reshaped_shape_const)
        if reshaped_size != in_shape_size:
          raise ValueError(
              "Cannot reshape a tensor with %d elements to shape %s "
              "(%d elements)." %
              (in_shape_size, original_reshaped_shape, reshaped_size))
        reshaped_shape = constant_op.constant(
            reshaped_shape_const, dtype=dtypes.int64)
    return sparse_tensor.SparseTensor(reshaped_ind,
                                      array_ops.identity(sp_input.values),
                                      reshaped_shape)
