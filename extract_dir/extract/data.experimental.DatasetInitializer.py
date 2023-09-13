@tf_export("data.experimental.DatasetInitializer")
class DatasetInitializer(lookup_ops.TableInitializerBase):
  """Creates a table initializer from a `tf.data.Dataset`.
  Sample usage:
  >>> keys = tf.data.Dataset.range(100)
  >>> values = tf.data.Dataset.range(100).map(
  ...     lambda x: tf.strings.as_string(x * 2))
  >>> ds = tf.data.Dataset.zip((keys, values))
  >>> init = tf.data.experimental.DatasetInitializer(ds)
  >>> table = tf.lookup.StaticHashTable(init, "")
  >>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
  array([b'0', b'2', b'4'], dtype=object)
  Attributes:
    dataset: A `tf.data.Dataset` object that produces tuples of scalars. The
      first scalar is treated as a key and the second as value.
  Raises: ValueError if `dataset` doesn't conform to specifications.
  """
  def __init__(self, dataset):
    """Creates a table initializer from a `tf.data.Dataset`.
    Args:
      dataset: A `tf.data.Dataset` object that produces tuples of scalars. The
        first scalar is treated as a key and the second as value.
    Raises: ValueError if `dataset` doesn't conform to specifications.
    Returns: A `DatasetInitializer` object
    """
    # Assert that the dataset element spec is a tuple of TensorSpecs where
    # each tensor is a scalar.
    self.dataset = dataset
    elem_spec = self.dataset.element_spec
    _check_table_initializer_element_spec(elem_spec)
    key_type = elem_spec[0].dtype
    value_type = elem_spec[1].dtype
    super(DatasetInitializer, self).__init__(key_type, value_type)
  def initialize(self, table):
    lookup_ops.check_table_dtypes(table, self._key_dtype, self._value_dtype)
    init_op = ged_ops.initialize_table_from_dataset(
        table.resource_handle, self.dataset._variant_tensor)  # pylint: disable=protected-access
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op
