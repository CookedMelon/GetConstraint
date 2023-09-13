@tf_export("tensordot", "linalg.tensordot")
@dispatch.add_dispatch_support
def tensordot(a, b, axes, name=None):
  r"""Tensor contraction of a and b along specified axes and outer product.
  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `axes`.
  This operation corresponds to `numpy.tensordot(a, b, axes)`.
  Example 1: When `a` and `b` are matrices (order 2), the case `axes=1`
  is equivalent to matrix multiplication.
  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.
  Example 3: When `a` and `b` are matrices (order 2), the case `axes=0` gives
  the outer product, a tensor of order 4.
  Example 4: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
  tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
  \\(c_{jklm}\\) whose entry
  corresponding to the indices \\((j,k,l,m)\\) is given by:
  \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
  In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
  Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
      If axes is a scalar, sum over the last N axes of a and the first N axes of
      b in order. If axes is a list or `Tensor` the first and second row contain
      the set of unique integers specifying axes along which the contraction is
      computed, for `a` and `b`, respectively. The number of axes for `a` and
      `b` must be equal. If `axes=0`, computes the outer product between `a` and
      `b`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same type as `a`.
  Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
      tensor.
  """
  def _tensordot_reshape(a, axes, flipped=False):
    """Helper method to perform transpose and reshape for contraction op.
    This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
    using `array_ops.transpose` and `array_ops.reshape`. The method takes a
    tensor and performs the correct transpose and reshape operation for a given
    set of indices. It returns the reshaped tensor as well as a list of indices
    necessary to reshape the tensor again after matrix multiplication.
    Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
        `a`.
      flipped: An optional `bool`. Defaults to `False`. If `True`, the method
        assumes that `a` is the second argument in the contraction operation.
    Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
      the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
      either a list of integers or an `int32` `Tensor`, depending on whether
      the shape of a is fully specified, and free_dims_static is either a list
      of integers and None values, or None, representing the inferred
      static shape of the free dimensions
    """
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in builtins.range(len(shape_a)) if i not in axes]
      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = list(axes) + free if flipped else free + list(axes)
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
      if (perm != np.arange(len(shape_a))).any():
        a_trans = array_ops.transpose(a, perm)
      else:
        a_trans = a
      if a_trans.get_shape().as_list() != new_shape:
        reshaped_a = array_ops.reshape(a_trans, new_shape)
      else:
        reshaped_a = a_trans
      return reshaped_a, free_dims, free_dims
    else:
      if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in builtins.range(len(shape_a)) if i not in axes]
        axes_dims = [shape_a[i] for i in axes]
        free_dims = [shape_a[i] for i in free]
        free_dims_static = free_dims
        axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
        free = ops.convert_to_tensor(free, dtype=dtypes.int32, name="free")
        shape_a = array_ops.shape(a)
      else:
        free_dims_static = None
        shape_a = array_ops.shape(a)
        rank_a = array_ops.rank(a)
        axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
        axes = array_ops.where(axes >= 0, axes, axes + rank_a)
        free, _ = gen_array_ops.list_diff(range(rank_a), axes, dtypes.int32)
      free_dims = array_ops.gather(shape_a, free)
      axes_dims = array_ops.gather(shape_a, axes)
      prod_free_dims = reduce_prod(free_dims)
      prod_axes_dims = reduce_prod(axes_dims)
      if flipped:
        perm = array_ops.concat([axes, free], 0)
        new_shape = array_ops_stack.stack([prod_axes_dims, prod_free_dims])
      else:
        perm = array_ops.concat([free, axes], 0)
        new_shape = array_ops_stack.stack([prod_free_dims, prod_axes_dims])
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims, free_dims_static
  def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, compat.integral_types):
      if axes < 0:
        raise ValueError(f"`axes` must be at least 0. Received: {axes}.")
      if a_shape.ndims is not None:
        if axes > a_shape.ndims:
          raise ValueError(f"`axes` must not be larger than the number of "
                           f"dimensions of tensor {a}.  Received {axes}, vs "
                           f"tensor dimensions {a_shape.ndims}.")
        return (list(builtins.range(a_shape.ndims - axes,
                                    a_shape.ndims)), list(builtins.range(axes)))
      else:
        rank = array_ops.rank(a)
        return (range(rank - axes, rank,
                      dtype=dtypes.int32), range(axes, dtype=dtypes.int32))
    elif isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError(
            f"`axes` must be an integer or have length 2. Received {axes}.")
      a_axes = axes[0]
      b_axes = axes[1]
      if isinstance(a_axes, compat.integral_types) and \
          isinstance(b_axes, compat.integral_types):
        a_axes = [a_axes]
        b_axes = [b_axes]
      if len(a_axes) != len(b_axes):
        raise ValueError(f"Different number of contraction axes `a` and `b`, "
                         f"{len(a_axes)} != {len(b_axes)}.")
      return a_axes, b_axes
    else:
      axes = ops.convert_to_tensor(axes, name="axes", dtype=dtypes.int32)
      return axes[0], axes[1]
  with ops.name_scope(name, "Tensordot", [a, b, axes]) as name:
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b")
    a_axes, b_axes = _tensordot_axes(a, axes)
    a_reshape, a_free_dims, a_free_dims_static = _tensordot_reshape(a, a_axes)
    b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
        b, b_axes, True)
    ab_matmul = matmul(a_reshape, b_reshape)
    if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
      if (ab_matmul.get_shape().is_fully_defined() and
          ab_matmul.get_shape().as_list() == a_free_dims + b_free_dims):
        return ab_matmul
      else:
        return array_ops.reshape(
            ab_matmul, a_free_dims + b_free_dims, name=name)
    else:
      a_free_dims = ops.convert_to_tensor(a_free_dims, dtype=dtypes.int32)
      b_free_dims = ops.convert_to_tensor(b_free_dims, dtype=dtypes.int32)
      product = array_ops.reshape(
          ab_matmul, array_ops.concat([a_free_dims, b_free_dims], 0), name=name)
      if a_free_dims_static is not None and b_free_dims_static is not None:
        product.set_shape(a_free_dims_static + b_free_dims_static)
      return product
