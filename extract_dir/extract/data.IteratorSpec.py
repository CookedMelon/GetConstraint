@tf_export("data.IteratorSpec", v1=[])
class IteratorSpec(type_spec.TypeSpec):
  """Type specification for `tf.data.Iterator`.
  For instance, `tf.data.IteratorSpec` can be used to define a tf.function that
  takes `tf.data.Iterator` as an input argument:
  >>> @tf.function(input_signature=[tf.data.IteratorSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def square(iterator):
  ...   x = iterator.get_next()
  ...   return x * x
  >>> dataset = tf.data.Dataset.from_tensors(5)
  >>> iterator = iter(dataset)
  >>> print(square(iterator))
  tf.Tensor(25, shape=(), dtype=int32)
  Attributes:
    element_spec: A (nested) structure of `tf.TypeSpec` objects that represents
      the type specification of the iterator elements.
  """
  __slots__ = ["_element_spec"]
  def __init__(self, element_spec):
    self._element_spec = element_spec
  @property
  def value_type(self):
    return OwnedIterator
  def _serialize(self):
    return (self._element_spec,)
  @property
  def _component_specs(self):
    return (tensor_spec.TensorSpec([], dtypes.resource),)
  def _to_components(self, value):
    return (value._iterator_resource,)  # pylint: disable=protected-access
  def _from_components(self, components):
    return OwnedIterator(
        dataset=None,
        components=components,
        element_spec=self._element_spec)
  @staticmethod
  def from_value(value):
    return IteratorSpec(value.element_spec)  # pylint: disable=protected-access
