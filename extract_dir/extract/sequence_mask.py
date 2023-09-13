@tf_export("sequence_mask")
@dispatch.add_dispatch_support
def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
  """Returns a mask tensor representing the first N positions of each cell.
  If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
  dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with
  ```
  mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
  ```
  Examples:
  ```python
  tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                  #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]
  tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                    #   [True, True, True]],
                                    #  [[True, True, False],
                                    #   [False, False, False]]]
  ```
  Args:
    lengths: integer tensor, all its values <= maxlen.
    maxlen: scalar integer tensor, size of last dimension of returned tensor.
      Default is the maximum value in `lengths`.
    dtype: output type of the resulting tensor.
    name: name of the op.
  Returns:
    A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
  Raises:
    ValueError: if `maxlen` is not a scalar.
  """
  with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
    lengths = ops.convert_to_tensor(lengths)
    if maxlen is None:
      maxlen = gen_math_ops._max(lengths, _all_dimensions(lengths))
      maxlen = gen_math_ops.maximum(constant(0, maxlen.dtype), maxlen)
    else:
      maxlen = ops.convert_to_tensor(maxlen)
    if maxlen.get_shape().ndims is not None and maxlen.get_shape().ndims != 0:
      raise ValueError("Argument `maxlen` must be scalar for sequence_mask, "
                       f"received `maxlen` = {maxlen} "
                       f"with shape '{maxlen.get_shape()}' instead")
    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = gen_math_ops._range(
        constant(0, maxlen.dtype), maxlen, constant(1, maxlen.dtype))
    # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
    # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
    matrix = gen_math_ops.cast(expand_dims(lengths, -1), maxlen.dtype)
    result = row_vector < matrix
    if dtype is None or result.dtype.is_compatible_with(dtype):
      return result
    else:
      return gen_math_ops.cast(result, dtype)
