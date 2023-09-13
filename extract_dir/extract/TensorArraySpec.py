@tf_export("TensorArraySpec")
@type_spec_registry.register("tf.TensorArraySpec")
class TensorArraySpec(type_spec.TypeSpec):
  """Type specification for a `tf.TensorArray`."""
  __slots__ = ["_element_shape", "_dtype", "_dynamic_size", "_infer_shape"]
  value_type = property(lambda self: TensorArray)
  def __init__(self,
               element_shape=None,
               dtype=dtypes.float32,
               dynamic_size=False,
               infer_shape=True):
    """Constructs a type specification for a `tf.TensorArray`.
    Args:
      element_shape: The shape of each element in the `TensorArray`.
      dtype: Data type of the `TensorArray`.
      dynamic_size: Whether the `TensorArray` can grow past its initial size.
      infer_shape: Whether shape inference is enabled.
    """
    self._element_shape = tensor_shape.as_shape(element_shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._dynamic_size = dynamic_size
    self._infer_shape = infer_shape
  def is_subtype_of(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, TensorArraySpec) and
            self._dtype == other._dtype and
            self._dynamic_size == other._dynamic_size)
  def most_specific_common_supertype(self, others):
    """Returns the most specific supertype of `self` and `others`.
    Args:
      others: A Sequence of `TypeSpec`.
    Returns `None` if a supertype does not exist.
    """
    # pylint: disable=protected-access
    if not all(isinstance(other, TensorArraySpec) for other in others):
      return False
    common_shape = self._element_shape.most_specific_common_supertype(
        other._element_shape for other in others)
    if common_shape is None:
      return None
    if not all(self._dtype == other._dtype for other in others):
      return None
    if not all(self._dynamic_size == other._dynamic_size for other in others):
      return None
    infer_shape = self._infer_shape and all(
        other._infer_shape for other in others)
    return TensorArraySpec(common_shape, self._dtype, self._dynamic_size,
                           infer_shape)
  def is_compatible_with(self, other):
    # pylint: disable=protected-access
    if not isinstance(other, type_spec.TypeSpec):
      other = type_spec.type_spec_from_value(other)
    # Note: we intentionally exclude infer_shape in this check.
    return (isinstance(other, TensorArraySpec) and
            self._dtype.is_compatible_with(other._dtype) and
            self._element_shape.is_compatible_with(other._element_shape) and
            self._dynamic_size == other._dynamic_size)
  def _serialize(self):
    return (self._element_shape, self._dtype, self._dynamic_size,
            self._infer_shape)
  @property
  def _component_specs(self):
    return [tensor_spec.TensorSpec([], dtypes.variant)]
  def _to_components(self, value):
    if not isinstance(value, TensorArray):
      raise TypeError("Expected value to be a TensorArray, but got: `{}`".format(
          type(value)))
    if value.flow is not None and value.flow.dtype == dtypes.variant:
      return [value.flow]
    else:
      # Convert to a TF2-style TensorArray.
      # TODO(ebrevdo): Add an "_as_variant" method to TensorArray class, or
      # "implementation / as_variant" arg to TensorArray constructor.
      with ops.name_scope("convert_tensor_array"):
        flow = list_ops.tensor_list_from_tensor(
            tensor=value.stack(), element_shape=value.element_shape)
      return [flow]
  def _from_components(self, tensor_list):
    # This will return a TF2 Graph-style TensorArray because tensor_list[0] is
    # a variant object.  size == -1 implies unknown size.
    ret = TensorArray(
        dtype=self._dtype,
        flow=tensor_list[0],
        dynamic_size=self._dynamic_size,
        infer_shape=self._infer_shape)
    ret._implementation._element_shape = [self._element_shape]  # pylint: disable=protected-access
    return ret
  @staticmethod
  def from_value(value):
    if not isinstance(value, TensorArray):
      raise TypeError("Expected value to be a TensorArray, but got: `{}`".format(
          type(value)))
    return TensorArraySpec(
        dtype=value.dtype,
        element_shape=value.element_shape,
        dynamic_size=value.dynamic_size,
        infer_shape=value._infer_shape)  # pylint: disable=protected-access
  def _to_legacy_output_types(self):
    return self._dtype
  def _to_legacy_output_shapes(self):
    # Sneak the dynamic_size and infer_shape values into the legacy shape.
    return (tensor_shape.TensorShape([self._dynamic_size, self._infer_shape
                                     ]).concatenate(self._element_shape))
  def _to_legacy_output_classes(self):
    return TensorArray
