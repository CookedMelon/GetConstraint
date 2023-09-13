@tf_export("lookup.StaticHashTable", v1=[])
class StaticHashTable(InitializableLookupTableBase):
  """A generic hash table that is immutable once initialized.
  Example usage:
  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table = tf.lookup.StaticHashTable(
  ...     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)
  Or for more pythonic code:
  >>> table[input_tensor].numpy()
  array([ 7, -1], dtype=int32)
  The result of a lookup operation has the same shape as the argument:
  >>> input_tensor = tf.constant([['a', 'b'], ['c', 'd']])
  >>> table[input_tensor].numpy()
  array([[ 7,  8],
         [ 9, -1]], dtype=int32)
  """
  def __init__(self,
               initializer,
               default_value,
               name=None,
               experimental_is_anonymous=False):
    """Creates a non-initialized `HashTable` object.
    Creates a table, the type of its keys and values are specified by the
    initializer.
    Before using the table you will have to initialize it. After initialization
    the table will be immutable.
    Args:
      initializer: The table initializer to use. See `HashTable` kernel for
        supported key and value types.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.
    Returns:
      A `HashTable` object.
    """
    self._initializer = initializer
    self._default_value = default_value
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = self._initializer._shared_name  # pylint: disable=protected-access
      if not self._shared_name:
        # Force using a shared name so that StaticHashTable resources can be
        # shared across different kernels. If no "shared_name" is set and
        # "use_node_name_sharing" is False, then each kernel gets its own local
        # resource.
        self._shared_name = "hash_table_%s" % (str(uuid.uuid4()),)
    self._name = name or "hash_table"
    self._table_name = None
    super(StaticHashTable, self).__init__(default_value, initializer)
    self._value_shape = self._default_value.get_shape()
  def _create_resource(self):
    if self._is_anonymous:
      table_ref = gen_lookup_ops.anonymous_hash_table(
          key_dtype=self._initializer.key_dtype,
          value_dtype=self._initializer.value_dtype,
          name=self._name)
    else:
      table_ref = gen_lookup_ops.hash_table_v2(
          shared_name=self._shared_name,
          key_dtype=self._initializer.key_dtype,
          value_dtype=self._initializer.value_dtype,
          name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref
  @property
  def name(self):
    return self._table_name
  def export(self, name=None):
    """Returns tensors of all keys and values in the table.
    Args:
      name: A name for the operation (optional).
    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_Export" % self.name, [self.resource_handle]):
      exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
          self.resource_handle, self._key_dtype, self._value_dtype)
    exported_values.set_shape(exported_keys.get_shape().concatenate(
        self._value_shape))
    return exported_keys, exported_values
  def _serialize_to_proto(self, **unused_kwargs):
    return None
  def _add_trackable_child(self, name, value):
    setattr(self, name, value)
    if isinstance(value, trackable_base.Trackable):
      self._track_trackable(value, name)  # pylint:disable=protected-access
  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    class _RestoredStaticHashTable(resource.RestoredResource):  # pylint: disable=protected-access
      @classmethod
      def _resource_type(cls):
        return "RestoredStaticHashTable"
    return _RestoredStaticHashTable._deserialize_from_proto(**kwargs)  # pylint: disable=protected-access
