@tf_export("IndexedSlicesSpec")
class IndexedSlicesSpec(type_spec.TypeSpec):
  """Type specification for a `tf.IndexedSlices`."""
  __slots__ = ["_shape", "_values_dtype", "_indices_dtype",
               "_dense_shape_dtype", "_indices_shape"]
  value_type = property(lambda self: IndexedSlices)
  def __init__(self, shape=None, dtype=dtypes.float32,
               indices_dtype=dtypes.int64, dense_shape_dtype=None,
               indices_shape=None):
    """Constructs a type specification for a `tf.IndexedSlices`.
    Args:
      shape: The dense shape of the `IndexedSlices`, or `None` to allow any
        dense shape.
      dtype: `tf.DType` of values in the `IndexedSlices`.
      indices_dtype: `tf.DType` of the `indices` in the `IndexedSlices`.  One
        of `tf.int32` or `tf.int64`.
      dense_shape_dtype: `tf.DType` of the `dense_shape` in the `IndexedSlices`.
        One of `tf.int32`, `tf.int64`, or `None` (if the `IndexedSlices` has
        no `dense_shape` tensor).
      indices_shape: The shape of the `indices` component, which indicates
        how many slices are in the `IndexedSlices`.
    """
    self._shape = tensor_shape.as_shape(shape)
    self._values_dtype = dtypes.as_dtype(dtype)
    self._indices_dtype = dtypes.as_dtype(indices_dtype)
    if dense_shape_dtype is None:
      self._dense_shape_dtype = None
    else:
      self._dense_shape_dtype = dtypes.as_dtype(dense_shape_dtype)
    self._indices_shape = tensor_shape.as_shape(indices_shape).with_rank(1)
  def _serialize(self):
    return (self._shape, self._values_dtype, self._indices_dtype,
            self._dense_shape_dtype, self._indices_shape)
  @property
  def _component_specs(self):
    value_shape = self._indices_shape.concatenate(self._shape[1:])
    specs = [
        tensor_spec.TensorSpec(value_shape, self._values_dtype),
        tensor_spec.TensorSpec(self._indices_shape, self._indices_dtype)]
    if self._dense_shape_dtype is not None:
      specs.append(
          tensor_spec.TensorSpec([self._shape.ndims], self._dense_shape_dtype))
    return tuple(specs)
  def _to_components(self, value):
    if value.dense_shape is None:
      return (value.values, value.indices)
    else:
      return (value.values, value.indices, value.dense_shape)
  def _from_components(self, tensor_list):
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      if len(tensor_list) == 2:
        return IndexedSlicesValue(tensor_list[0], tensor_list[1], None)
      else:
        return IndexedSlicesValue(*tensor_list)
    else:
      return IndexedSlices(*tensor_list)
