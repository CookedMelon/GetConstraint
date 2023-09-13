@tf_export("experimental.Optional", "data.experimental.Optional")
@deprecation.deprecated_endpoints("data.experimental.Optional")
class Optional(composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
  """Represents a value that may or may not be present.
  A `tf.experimental.Optional` can represent the result of an operation that may
  fail as a value, rather than raising an exception and halting execution. For
  example, `tf.data.Iterator.get_next_as_optional()` returns a
  `tf.experimental.Optional` that either contains the next element of an
  iterator if one exists, or an "empty" value that indicates the end of the
  sequence has been reached.
  `tf.experimental.Optional` can only be used with values that are convertible
  to `tf.Tensor` or `tf.CompositeTensor`.
  One can create a `tf.experimental.Optional` from a value using the
  `from_value()` method:
  >>> optional = tf.experimental.Optional.from_value(42)
  >>> print(optional.has_value())
  tf.Tensor(True, shape=(), dtype=bool)
  >>> print(optional.get_value())
  tf.Tensor(42, shape=(), dtype=int32)
  or without a value using the `empty()` method:
  >>> optional = tf.experimental.Optional.empty(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))
  >>> print(optional.has_value())
  tf.Tensor(False, shape=(), dtype=bool)
  """
  @abc.abstractmethod
  def has_value(self, name=None):
    """Returns a tensor that evaluates to `True` if this optional has a value.
    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    Args:
      name: (Optional.) A name for the created operation.
    Returns:
      A scalar `tf.Tensor` of type `tf.bool`.
    """
    raise NotImplementedError("Optional.has_value()")
  @abc.abstractmethod
  def get_value(self, name=None):
    """Returns the value wrapped by this optional.
    If this optional does not have a value (i.e. `self.has_value()` evaluates to
    `False`), this operation will raise `tf.errors.InvalidArgumentError` at
    runtime.
    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)
    Args:
      name: (Optional.) A name for the created operation.
    Returns:
      The wrapped value.
    """
    raise NotImplementedError("Optional.get_value()")
  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this optional.
    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.element_spec)
    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this optional, specifying the type of individual components.
    """
    raise NotImplementedError("Optional.element_spec")
  @staticmethod
  def empty(element_spec):
    """Returns an `Optional` that has no value.
    NOTE: This method takes an argument that defines the structure of the value
    that would be contained in the returned `Optional` if it had a value.
    >>> optional = tf.experimental.Optional.empty(
    ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))
    >>> print(optional.has_value())
    tf.Tensor(False, shape=(), dtype=bool)
    Args:
      element_spec: A (nested) structure of `tf.TypeSpec` objects matching the
        structure of an element of this optional.
    Returns:
      A `tf.experimental.Optional` with no value.
    """
    return _OptionalImpl(gen_optional_ops.optional_none(), element_spec)
  @staticmethod
  def from_value(value):
    """Returns a `tf.experimental.Optional` that wraps the given value.
    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)
    Args:
      value: A value to wrap. The value must be convertible to `tf.Tensor` or
        `tf.CompositeTensor`.
    Returns:
      A `tf.experimental.Optional` that wraps `value`.
    """
    with ops.name_scope("optional") as scope:
      with ops.name_scope("value"):
        element_spec = structure.type_spec_from_value(value)
        encoded_value = structure.to_tensor_list(element_spec, value)
    return _OptionalImpl(
        gen_optional_ops.optional_from_value(encoded_value, name=scope),
        element_spec,
    )
