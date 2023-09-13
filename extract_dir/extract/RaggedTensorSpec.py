@tf_export("RaggedTensorSpec")
@type_spec_registry.register("tf.RaggedTensorSpec")
class RaggedTensorSpec(type_spec.BatchableTypeSpec):
  """Type specification for a `tf.RaggedTensor`."""
  __slots__ = [
      "_shape", "_dtype", "_ragged_rank", "_row_splits_dtype",
      "_flat_values_spec"
  ]
  @property
  def dtype(self):
    """The `tf.dtypes.DType` specified by this type for the RaggedTensor.
    Examples:
    >>> rt = tf.ragged.constant([["a"], ["b", "c"]], dtype=tf.string)
    >>> tf.type_spec_from_value(rt).dtype
    tf.string
    Returns:
      A `tf.dtypes.DType` of the values in the RaggedTensor.
    """
    return self._dtype
  @property
  def shape(self):
    """The statically known shape of the RaggedTensor.
    Examples:
    >>> rt = tf.ragged.constant([[0], [1, 2]])
    >>> tf.type_spec_from_value(rt).shape
    TensorShape([2, None])
    >>> rt = tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1)
    >>> tf.type_spec_from_value(rt).shape
    TensorShape([2, None, 2])
    Returns:
      A `tf.TensorShape` containing the statically known shape of the
      RaggedTensor. Ragged dimensions have a size of `None`.
    """
    return self._shape
  @property
  def ragged_rank(self):
    """The number of times the RaggedTensor's flat_values is partitioned.
    Defaults to `shape.ndims - 1`.
    Examples:
    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    >>> tf.type_spec_from_value(values).ragged_rank
    1
    >>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)
    >>> tf.type_spec_from_value(rt1).ragged_rank
    2
    Returns:
      A Python `int` indicating the number of times the underlying `flat_values`
      Tensor has been partitioned to add a new dimension.
      I.e., `tf.rank(rt) = tf.rank(rt.flat_values) + rt.ragged_rank`.
    """
    return self._ragged_rank
  @property
  def row_splits_dtype(self):
    """The `tf.dtypes.DType` of the RaggedTensor's `row_splits`.
    Examples:
    >>> rt = tf.ragged.constant([[1, 2, 3], [4]], row_splits_dtype=tf.int64)
    >>> tf.type_spec_from_value(rt).row_splits_dtype
    tf.int64
    Returns:
      A `tf.dtypes.DType` for the RaggedTensor's `row_splits` tensor. One
      of `tf.int32` or `tf.int64`.
    """
    return self._row_splits_dtype
  @property
  def flat_values_spec(self):
    """The `TypeSpec` of the flat_values of RaggedTensor.
    Returns:
      - The TypeSpec of flat_values.
      - None when the flat_values is a Tensor.
    """
    return self._flat_values_spec
  @property
  def value_type(self):
    return RaggedTensor if self._ragged_rank > 0 else ops.Tensor
  def __init__(self,
               shape=None,
               dtype=dtypes.float32,
               ragged_rank=None,
               row_splits_dtype=dtypes.int64,
               flat_values_spec=None):
    """Constructs a type specification for a `tf.RaggedTensor`.
    Args:
      shape: The shape of the RaggedTensor, or `None` to allow any shape.  If a
        shape is specified, then all ragged dimensions must have size `None`.
      dtype: `tf.DType` of values in the RaggedTensor.
      ragged_rank: Python integer, the number of times the RaggedTensor's
        flat_values is partitioned.  Defaults to `shape.ndims - 1`.
      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One
        of `tf.int32` or `tf.int64`.
      flat_values_spec: TypeSpec for flat_value of the RaggedTensor. It shall be
        provided when the flat_values is a CompositeTensor rather then Tensor.
        If both `dtype` and `flat_values_spec` and  are provided, `dtype` must
        be the same as `flat_values_spec.dtype`. (experimental)
    """
    self._shape = tensor_shape.as_shape(shape)
    self._row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if flat_values_spec is not None:
      if dtype is None:
        dtype = flat_values_spec.dtype
      elif dtype != flat_values_spec.dtype:
        raise ValueError("dtype must be the same as flat_values_spec.dtype")
    elif dtype is None:
      raise ValueError(
          "At least one of dtype or flat_values_spec must be provided")
    self._dtype = dtypes.as_dtype(dtype)
    self._flat_values_spec = flat_values_spec
    rank = self._shape.ndims
    if ragged_rank is None:
      if rank is None:
        raise ValueError("Must specify ragged_rank or "
                         "a shape with a known rank.")
      ragged_rank = rank - 1
    self._ragged_rank = ragged_rank
    if not isinstance(self._ragged_rank, int):
      raise TypeError(f"Argument `ragged_rank` must be an int. "
                      f"Received {ragged_rank}.")
    if rank is not None:
      if ragged_rank >= rank:
        raise ValueError(f"Argument `ragged_rank` ({ragged_rank}) must be less "
                         f"than rank ({rank}).")
  def is_compatible_with(self, spec_or_value):
    # RaggedTensor with ragged_rank 0 can be compatible with raw flat_values.
    if self._ragged_rank == 0:
      if self._flat_values_spec is None:
        if isinstance(spec_or_value, (ops.Tensor, tensor_spec.TensorSpec)):
          return tensor_spec.TensorSpec(
              self._shape, self._dtype).is_compatible_with(spec_or_value)
      elif not isinstance(spec_or_value, (RaggedTensor, RaggedTensorSpec)):
        return self._flat_values_spec.is_compatible_with(spec_or_value)
    return super(RaggedTensorSpec, self).is_compatible_with(spec_or_value)
  def _serialize(self):
    if self._flat_values_spec is None:
      return (self._shape, self._dtype, self._ragged_rank,
              self._row_splits_dtype)
    else:
      return (self._shape, self._dtype, self._ragged_rank,
              self._row_splits_dtype, self._flat_values_spec)
  @property
  def _component_specs(self):
    if self._ragged_rank <= 0:
      if self._flat_values_spec is not None:
        return [self._flat_values_spec]
      else:
        return [tensor_spec.TensorSpec(self._shape, self._dtype)]
    flat_values_spec = self._flat_values_spec
    if flat_values_spec is None:
      flat_values_shape = tensor_shape.TensorShape([None]).concatenate(
          self._shape[self._ragged_rank + 1:])
      flat_values_spec = tensor_spec.TensorSpec(flat_values_shape, self._dtype)
    outer_dim = tensor_shape.dimension_at_index(self._shape, 0)
    outer_splits_shape = [None if outer_dim is None else outer_dim + 1]
    inner_splits_spec = tensor_spec.TensorSpec([None], self._row_splits_dtype)
    specs = ([
        flat_values_spec,
        tensor_spec.TensorSpec(outer_splits_shape, self._row_splits_dtype)
    ] + [inner_splits_spec for _ in range(self._ragged_rank - 1)])
    return specs
  def _to_components(self, value):
    if is_ragged(value):
      return [value.flat_values] + list(value.nested_row_splits)
    else:
      return [value]
  def _from_components(self, tensor_list):
    result = tensor_list[0]
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      for row_splits in reversed(tensor_list[1:]):
        result = ragged_tensor_value.RaggedTensorValue(result, row_splits)
    else:
      if isinstance(tensor_list[0], np.ndarray):
        tensor_list = [ops.convert_to_tensor(t) for t in tensor_list]
        result = tensor_list[0]
      for row_splits in reversed(tensor_list[1:]):
        result = RaggedTensor(
            result,
            RowPartition.from_row_splits(row_splits, validate=False),
            internal=True)
    if self._shape.ndims is not None:
      if isinstance(result, RaggedTensor):
        result._set_shape(self._shape)  # pylint: disable=protected-access
        # TODO(xjun): MaskedTensor doesn't implement set_shape.
        if self.flat_values_spec is not None and hasattr(result.flat_values,
                                                         "set_shape"):
          result.flat_values.set_shape(self.flat_values_spec.shape)
      elif isinstance(result, ops.Tensor):
        result.set_shape(self._shape)
    return result
  # The RaggedTensorSpec tensor_list encoding uses to/from_variant ops
  # to (un)box the component tensors in a way that allows for batching &
  # unbatching.
  @property
  def _flat_tensor_specs(self):
    # NOTE(mishragaurav): The default flat shape of a boxed `RaggedTensor` is
    # `[]` (scalar), but a `RaggedTensorSpec` can also represent a batch of
    # boxed `RaggedTensor` objects with shape `(...)` (and batches of batches,
    # etc.), so the flat shape must be unknown.
    return [tensor_spec.TensorSpec(None, dtypes.variant)]
  def _to_tensor_list(self, value):
    # TODO(edloper): Update gen_ragged_conversion_ops that convert to and
    # from variant to include all of the row-partitioning tensors.
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    if isinstance(value, RaggedTensor):
      if value.ragged_rank != self._ragged_rank:
        raise ValueError(
            f"Ragged rank of value {value.ragged_rank} does not match "
            f"ragged rank of type {self._ragged_rank}.")
      # pylint: disable=protected-access
      return [value._to_variant(batched_input=False)]
    else:
      if self._ragged_rank > 0:
        raise ValueError(
            f"Expected a RaggedTensor if ragged rank={self._ragged_rank}"
            f" but got {type(value).__name__}."
        )
      return [
          gen_ragged_conversion_ops.ragged_tensor_to_variant(
              (), value, batched_input=False)
      ]
  def _to_batched_tensor_list(self, value):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    if isinstance(value, RaggedTensor):
      if value.ragged_rank != self._ragged_rank:
        raise ValueError(
            f"Ragged rank of value {value.ragged_rank} does not match "
            f"ragged rank of type {self._ragged_rank}.")
      # pylint: disable=protected-access
      return [value._to_variant(batched_input=True)]
    else:
      if self._ragged_rank > 0:
        raise ValueError(
            f"Expected a RaggedTensor if ragged rank={self._ragged_rank}"
            f" but got {type(value).__name__}."
        )
      return [
          gen_ragged_conversion_ops.ragged_tensor_to_variant(
              rt_nested_splits=(), rt_dense_values=value, batched_input=True)
      ]
  def _from_compatible_tensor_list(self, tensor_list):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    result = RaggedTensor._from_variant(  # pylint: disable=protected-access
        tensor_list[0],
        dtype=self._dtype,
        row_splits_dtype=self._row_splits_dtype,
        output_ragged_rank=self._ragged_rank)
    if self._shape.ndims is not None:
      if isinstance(result, RaggedTensor):
        result._set_shape(self._shape)  # pylint: disable=protected-access
        # TODO(xjun): MaskedTensor doesn't implement set_shape.
        if self.flat_values_spec is not None and hasattr(self.flat_values,
                                                         "set_shape"):
          result.flat_values.set_shape(self.flat_values_spec.shape)
      else:
        result.set_shape(self._shape)
    return result
  def _batch(self, batch_size):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    return RaggedTensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype, self._ragged_rank + 1, self._row_splits_dtype)
  def _unbatch(self):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    # Note: Negative ragged_rank is allowed here because the dataset could be
    # subsequently batched again. If ragged_rank > 1, assume row_splits_dtype is
    # consistent. Errors are handled in
    # RaggedTensorSpec._from_compatible_tensor_list()
    return RaggedTensorSpec(self._shape[1:], self._dtype, self._ragged_rank - 1,
                            self._row_splits_dtype)
  def _to_legacy_output_types(self):
    return self._dtype
  def _to_legacy_output_shapes(self):
    return self._shape
  def _to_legacy_output_classes(self):
    return self
  @classmethod
  def from_value(cls, value):
    if (isinstance(value, ragged_tensor_value.RaggedTensorValue) or
        isinstance(value.flat_values, ops.Tensor)):
      return cls(
          shape=value.shape,
          dtype=value.values.dtype,
          ragged_rank=value.ragged_rank,
          row_splits_dtype=value.row_splits.dtype)
    else:
      flat_values_spec = type_spec.type_spec_from_value(value.flat_values)
      # Relax shape[0] to None, as it is connected to dynamic ragged shapes.
      flat_values_spec = flat_values_spec._unbatch()._batch(None)  # pylint: disable=protected-access
      return cls(
          shape=value.shape,
          dtype=value.values.dtype,
          ragged_rank=value.ragged_rank,
          row_splits_dtype=value.row_splits.dtype,
          flat_values_spec=flat_values_spec)
