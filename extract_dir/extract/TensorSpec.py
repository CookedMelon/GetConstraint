@tf_export("TensorSpec")
@type_spec_registry.register("tf.TensorSpec")
class TensorSpec(DenseSpec, type_spec.BatchableTypeSpec,
                 trace_type.Serializable, internal.TensorSpec):
  """Describes the type of a tf.Tensor.
  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> tf.TensorSpec.from_tensor(t)
  TensorSpec(shape=(2, 3), dtype=tf.int32, name=None)
  Contains metadata for describing the the nature of `tf.Tensor` objects
  accepted or returned by some TensorFlow APIs.
  For example, it can be used to constrain the type of inputs accepted by
  a tf.function:
  >>> @tf.function(input_signature=[tf.TensorSpec([1, None])])
  ... def constrained_foo(t):
  ...   print("tracing...")
  ...   return t
  Now the `tf.function` is able to assume that `t` is always of the type
  `tf.TensorSpec([1, None])` which will avoid retracing as well as enforce the
  type restriction on inputs.
  As a result, the following call with tensor of type `tf.TensorSpec([1, 2])`
  triggers a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2]])).numpy()
  tracing...
  array([[1., 2.]], dtype=float32)
  The following subsequent call with tensor of type `tf.TensorSpec([1, 4])`
  does not trigger a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2, 3, 4]])).numpy()
  array([[1., 2., 3., 4.], dtype=float32)
  But the following call with tensor of type `tf.TensorSpec([2, 2])` fails:
  >>> constrained_foo(tf.constant([[1., 2], [3, 4]])).numpy()
  Traceback (most recent call last):
  ...
  TypeError: Binding inputs to tf.function `constrained_foo` failed ...
  """
  __slots__ = []
  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.TensorSpecProto]:
    """Returns the type of proto associated with TensorSpec serialization."""
    return struct_pb2.TensorSpecProto
  @classmethod
  def experimental_from_proto(
      cls, proto: struct_pb2.TensorSpecProto) -> "TensorSpec":
    """Returns a TensorSpec instance based on the serialized proto."""
    return TensorSpec(
        shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape),
        dtype=proto.dtype,
        name=proto.name if proto.name else None)
  def experimental_as_proto(self) -> struct_pb2.TensorSpecProto:
    """Returns a proto representation of the TensorSpec instance."""
    return struct_pb2.TensorSpecProto(
        shape=self.shape.experimental_as_proto(),
        dtype=self.dtype.experimental_as_proto().datatype,
        name=self.name)
  def is_compatible_with(self, spec_or_tensor):  # pylint:disable=useless-super-delegation,arguments-renamed
    """Returns True if spec_or_tensor is compatible with this TensorSpec.
    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).
    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor
    Returns:
      True if spec_or_tensor is compatible with self.
    """
    return super(TensorSpec, self).is_compatible_with(spec_or_tensor)
  def is_subtype_of(self, other):
    if not isinstance(other, TensorSpec):
      return False
    return (
        (not self.name or self.name == other.name)
        and self.shape.is_subtype_of(other.shape)
        and self.dtype.is_subtype_of(other.dtype)
    )
  def placeholder_value(self, placeholder_context):
    """Generates a graph_placholder with the given TensorSpec information."""
    if placeholder_context.unnest_only:
      return self
    name = self.name or placeholder_context.naming_scope
    context_graph = placeholder_context.context_graph
    if placeholder_context.with_none_control_dependencies:
      # Note: setting ops.control_dependencies(None) ensures we always put
      # capturing placeholders outside of any control flow context.
      with context_graph.control_dependencies(None):
        placeholder = self._graph_placeholder(context_graph, name=name)
    else:
      placeholder = self._graph_placeholder(context_graph, name=name)
    if name is not None:
      # Record the requested/user-specified name in case it's different than
      # the uniquified name, for validation when exporting signatures.
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_user_specified_name",
          attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
    handle_data = self.dtype._handle_data  # pylint: disable=protected-access
    if (
        handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type
    ):
      handle_data_util.set_handle_data(placeholder, handle_data)
    # Record the composite device as an attribute to the placeholder.
    # This attribute would be propagated into the arg_attr of the FunctionDef.
    # Currently, a packed eager tensor is always placed on a CompositeDevice.
    if placeholder_context.composite_device_name is not None:
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_composite_device",
          attr_value_pb2.AttrValue(s=compat.as_bytes(
              placeholder_context.composite_device_name)))
    return placeholder
  def _graph_placeholder(self, graph, name=None):
    """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
    dtype = self.dtype.base_dtype
    shape = self.shape
    dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    if isinstance(shape, (list, tuple)):
      shape = tensor_shape.TensorShape(shape)
    shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
    attrs = {"dtype": dtype_value, "shape": shape}
    try:
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[],
          attrs=attrs, name=name)
    except ValueError as e:
      # TODO(b/262413656) Sometimes parameter names are not valid op names, in
      # which case an unnamed placeholder is created instead. Update this logic
      # to sanitize the name instead of falling back on unnamed placeholders.
      logging.warning(e)
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[], attrs=attrs)
    (result,) = op.outputs
    if op_callbacks.should_invoke_op_callbacks():
      # TODO(b/147670703): Once the special-op creation code paths
      # are unified. Remove this `if` block.
      callback_outputs = op_callbacks.invoke_op_callbacks(
          "Placeholder", tuple(), attrs, tuple(op.outputs),
          op_name=name, graph=graph)
      if callback_outputs is not None:
        (result,) = callback_outputs
    return result
  def _to_tensors(self, value):
    assert isinstance(value, ops.Tensor)
    return [value]
  def _flatten(self):
    return [self]
  def _cast(self, value, casting_context):
    """Cast value to a tensor that is a subtype of this TensorSpec."""
    # This method is mainly used to cast Python primitives to tensor.
    # Currently, cast tensor to tensor with different types are not supported.
    # For example, casting int32 to float32 would raise a ValueError.
    if casting_context.allow_specs and isinstance(value, TensorSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self
    value = ops.convert_to_tensor(value, self.dtype)
    value_spec = TensorSpec(value.shape, value.dtype, self.name)
    if not value_spec.is_subtype_of(self):
      if self.is_subtype_of(value_spec):
        gen_array_ops.ensure_shape(value, self.shape)
      else:
        raise AssertionError(f"Can not cast {value_spec!r} to {self!r}")
    return value
  @classmethod
  def from_spec(cls, spec, name=None):
    """Returns a `TensorSpec` with the same shape and dtype as `spec`.
    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="OriginalName")
    >>> tf.TensorSpec.from_spec(spec, "NewName")
    TensorSpec(shape=(8, 3), dtype=tf.int32, name='NewName')
    Args:
      spec: The `TypeSpec` used to create the new `TensorSpec`.
      name: The name for the new `TensorSpec`.  Defaults to `spec.name`.
    """
    return cls(spec.shape, spec.dtype, name or spec.name)
  @classmethod
  def from_tensor(cls, tensor, name=None):
    """Returns a `TensorSpec` that describes `tensor`.
    >>> tf.TensorSpec.from_tensor(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)
    Args:
      tensor: The `tf.Tensor` that should be described.
      name: A name for the `TensorSpec`.  Defaults to `tensor.op.name`.
    Returns:
      A `TensorSpec` that describes `tensor`.
    """
    if isinstance(tensor, ops.EagerTensor):
      return TensorSpec(tensor.shape, tensor.dtype, name)
    elif isinstance(tensor, ops.Tensor):
      # TODO(b/249802365): Return a sanitized version of op name or no name.
      return TensorSpec(tensor.shape, tensor.dtype, name or tensor.op.name)
    else:
      raise ValueError(
          f"`tensor` should be a tf.Tensor, but got type {type(tensor)}.")
  @property
  def value_type(self):
    """The Python type for values that are compatible with this TypeSpec."""
    return ops.Tensor
  def _to_components(self, value):
    assert isinstance(value, core_tf_types.Tensor)
    return value
  def _from_components(self, components):
    return components
  def _from_compatible_tensor_list(self, tensor_list):
    # TODO(b/112266545): It would be cleaner to create a new `ensure_shape()`
    # op here and return that, instead of mutating the input's shape using
    # `Tensor.set_shape()`. However, that would add extra ops, which could
    # impact performance. When this bug is resolved, we should be able to add
    # the `ensure_shape()` ops and optimize them away using contextual shape
    # information.
    assert len(tensor_list) == 1
    tensor_list[0].set_shape(self._shape)
    return tensor_list[0]
  def _to_batchable_tensor_list(self, value, batched=False):
    if batched and self._shape.merge_with(value.shape).ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return self._to_components(value)
  def _batch(self, batch_size):
    return TensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype)
  def _unbatch(self):
    if self._shape.ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return TensorSpec(self._shape[1:], self._dtype)
  @property
  def _flat_tensor_specs(self):
    return [self]
  def _to_tensor_list(self, value):
    return [self._to_components(value)]
  def _to_batched_tensor_list(self, value):
    return self._to_tensor_list(value)
  # TODO(b/206014848): Helper function to support logic that does not consider
  # Tensor name. Will be removed once load-bearing usages of Tensor name are
  # fixed.
  def _without_tensor_names(self) -> "TensorSpec":
    """Returns a version of `TensorSpec` with the name removed."""
    if self.name is None:
      return self
    else:
      return TensorSpec(self.shape, self.dtype)
