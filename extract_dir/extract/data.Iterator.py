@tf_export("data.Iterator", v1=[])
class IteratorBase(
    collections_abc.Iterator,
    trackable.Trackable,
    composite_tensor.CompositeTensor,
    metaclass=abc.ABCMeta):
  """Represents an iterator of a `tf.data.Dataset`.
  `tf.data.Iterator` is the primary mechanism for enumerating elements of a
  `tf.data.Dataset`. It supports the Python Iterator protocol, which means
  it can be iterated over using a for-loop:
  >>> dataset = tf.data.Dataset.range(2)
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(0, shape=(), dtype=int64)
  tf.Tensor(1, shape=(), dtype=int64)
  or by fetching individual elements explicitly via `get_next()`:
  >>> dataset = tf.data.Dataset.range(2)
  >>> iterator = iter(dataset)
  >>> print(iterator.get_next())
  tf.Tensor(0, shape=(), dtype=int64)
  >>> print(iterator.get_next())
  tf.Tensor(1, shape=(), dtype=int64)
  In addition, non-raising iteration is supported via `get_next_as_optional()`,
  which returns the next element (if available) wrapped in a
  `tf.experimental.Optional`.
  >>> dataset = tf.data.Dataset.from_tensors(42)
  >>> iterator = iter(dataset)
  >>> optional = iterator.get_next_as_optional()
  >>> print(optional.has_value())
  tf.Tensor(True, shape=(), dtype=bool)
  >>> optional = iterator.get_next_as_optional()
  >>> print(optional.has_value())
  tf.Tensor(False, shape=(), dtype=bool)
  """
  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this iterator.
    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> iterator.element_spec
    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).
    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator, specifying the type of individual components.
    """
    raise NotImplementedError("Iterator.element_spec")
  @abc.abstractmethod
  def get_next(self):
    """Returns the next element.
    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> print(iterator.get_next())
    tf.Tensor(42, shape=(), dtype=int32)
    Returns:
      A (nested) structure of values matching `tf.data.Iterator.element_spec`.
    Raises:
      `tf.errors.OutOfRangeError`: If the end of the iterator has been reached.
    """
    raise NotImplementedError("Iterator.get_next()")
  @abc.abstractmethod
  def get_next_as_optional(self):
    """Returns the next element wrapped in `tf.experimental.Optional`.
    If the iterator has reached the end of the sequence, the returned
    `tf.experimental.Optional` will have no value.
    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> optional = iterator.get_next_as_optional()
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)
    >>> optional = iterator.get_next_as_optional()
    >>> print(optional.has_value())
    tf.Tensor(False, shape=(), dtype=bool)
    Returns:
      A `tf.experimental.Optional` object representing the next element.
    """
    raise NotImplementedError("Iterator.get_next_as_optional()")
