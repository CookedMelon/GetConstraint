@tf_export("required_space_to_batch_paddings")
def required_space_to_batch_paddings(input_shape,
                                     block_shape,
                                     base_paddings=None,
                                     name=None):
  """Calculate padding required to make block_shape divide input_shape.
  This function can be used to calculate a suitable paddings argument for use
  with space_to_batch_nd and batch_to_space_nd.
  Args:
    input_shape: int32 Tensor of shape [N].
    block_shape: int32 Tensor of shape [N].
    base_paddings: Optional int32 Tensor of shape [N, 2].  Specifies the minimum
      amount of padding to use.  All elements must be >= 0.  If not specified,
      defaults to 0.
    name: string.  Optional name prefix.
  Returns:
    (paddings, crops), where:
    `paddings` and `crops` are int32 Tensors of rank 2 and shape [N, 2]
    satisfying:
        paddings[i, 0] = base_paddings[i, 0].
        0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
        (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0
        crops[i, 0] = 0
        crops[i, 1] = paddings[i, 1] - base_paddings[i, 1]
  Raises: ValueError if called with incompatible shapes.
  """
  with ops.name_scope(name, "required_space_to_batch_paddings",
                      [input_shape, block_shape]):
    input_shape = ops.convert_to_tensor(
        input_shape, dtype=dtypes.int32, name="input_shape")
    block_shape = ops.convert_to_tensor(
        block_shape, dtype=dtypes.int32, name="block_shape")
    block_shape.get_shape().assert_is_fully_defined()
    block_shape.get_shape().assert_has_rank(1)
    num_block_dims = block_shape.get_shape().dims[0].value
    if num_block_dims == 0:
      return zeros([0, 2], dtypes.int32), zeros([0, 2], dtypes.int32)
    input_shape.get_shape().assert_is_compatible_with([num_block_dims])
    if base_paddings is not None:
      base_paddings = ops.convert_to_tensor(
          base_paddings, dtype=dtypes.int32, name="base_paddings")
      base_paddings.get_shape().assert_is_compatible_with([num_block_dims, 2])
    else:
      base_paddings = zeros([num_block_dims, 2], dtypes.int32)
    const_block_shape = tensor_util.constant_value(block_shape)
    const_input_shape = tensor_util.constant_value(input_shape)
    const_base_paddings = tensor_util.constant_value(base_paddings)
    if (const_block_shape is not None and const_input_shape is not None and
        const_base_paddings is not None):
      block_shape = const_block_shape
      input_shape = const_input_shape
      base_paddings = const_base_paddings
    # Use same expression for both constant and non-constant case.
    pad_start = base_paddings[:, 0]
    orig_pad_end = base_paddings[:, 1]
    full_input_shape = input_shape + pad_start + orig_pad_end
    pad_end_extra = (block_shape - full_input_shape % block_shape) % block_shape
    pad_end = orig_pad_end + pad_end_extra
    result_paddings = array_ops_stack.stack(
        [[pad_start[i], pad_end[i]] for i in range(num_block_dims)],
        name="paddings")
    result_crops = array_ops_stack.stack(
        [[0, pad_end_extra[i]] for i in range(num_block_dims)], name="crops")
    return result_paddings, result_crops
