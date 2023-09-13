@tf_export("lookup.experimental.MutableHashTable")
@saveable_compat.legacy_saveable_name("table")
class MutableHashTable(LookupInterface):
  """A generic mutable hash table implementation.
  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.
  `MutableHashTable` requires additional memory during checkpointing and restore
  operations to create temporary key and value tensors.
  Example usage:
  >>> table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.string,
  ...                                                 value_dtype=tf.int64,
  ...                                                 default_value=-1)
  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9], dtype=tf.int64)
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table.insert(keys_tensor, vals_tensor)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1])
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(keys_tensor).numpy()
  array([ 7, 8, -1])
  >>> sorted(table.export()[0].numpy())
  [b'a', b'b']
  >>> sorted(table.export()[1].numpy())
  [7, 8]
  """
  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               name="MutableHashTable",
               checkpoint=True,
               experimental_is_anonymous=False):
    """Creates an empty `MutableHashTable` object.
    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.
    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.
    Returns:
      A `MutableHashTable` object.
    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = None
      if context.executing_eagerly():
        # TODO(allenl): This will leak memory due to kernel caching by
        # the shared_name attribute value (but is better than the
        # alternative of sharing everything by default when executing
        # eagerly; hopefully creating tables in a loop is uncommon).
        self._shared_name = "table_%d" % (ops.uid(),)
    super(MutableHashTable, self).__init__(key_dtype, value_dtype)
    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = MutableHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
  def _create_resource(self):
    if self._is_anonymous:
      if self._default_value.get_shape().ndims == 0:
        table_ref = gen_lookup_ops.anonymous_mutable_hash_table(
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            name=self._name)
      else:
        table_ref = gen_lookup_ops.anonymous_mutable_hash_table_of_tensors(
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            value_shape=self._default_value.get_shape(),
            name=self._name)
    else:
      # The table must be shared if checkpointing is requested for multi-worker
      # training to work correctly. Use the node name if no shared_name has been
      # explicitly specified.
      use_node_name_sharing = self._checkpoint and self._shared_name is None
      if self._default_value.get_shape().ndims == 0:
        table_ref = gen_lookup_ops.mutable_hash_table_v2(
            shared_name=self._shared_name,
            use_node_name_sharing=use_node_name_sharing,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            name=self._name)
      else:
        table_ref = gen_lookup_ops.mutable_hash_table_of_tensors_v2(
            shared_name=self._shared_name,
            use_node_name_sharing=use_node_name_sharing,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            value_shape=self._default_value.get_shape(),
            name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref
  @property
  def name(self):
    return self._table_name
  def size(self, name=None):
    """Compute the number of elements in this table.
    Args:
      name: A name for the operation (optional).
    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)
  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the table.
    If a key is not present in the table, it is silently ignored.
    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).
    Returns:
      The created Operation.
    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")
    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)
    return op
  def lookup(self, keys, dynamic_default_values=None, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.
    The `default_value` is used for keys not present in the table.
    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      dynamic_default_values: The values to use if a key is missing in the
        table. If None (by default), the `table.default_value` will be used.
        Shape of `dynamic_default_values` must be same with
        `table.default_value` or the lookup result tensor.
        In the latter case, each key will have a different default value.
        For example:
          ```python
          keys = [0, 1, 3]
          dynamic_default_values = [[1, 3, 4], [2, 3, 9], [8, 3, 0]]
          # The key '0' will use [1, 3, 4] as default value.
          # The key '1' will use [2, 3, 9] as default value.
          # The key '3' will use [8, 3, 0] as default value.
          ```
      name: A name for the operation (optional).
    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.
    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(
            self.resource_handle, keys, dynamic_default_values
            if dynamic_default_values is not None else self._default_value)
    return values
  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.
    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).
    Returns:
      The created Operation.
    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self.name,
                        [self.resource_handle, keys, values]):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
    return op
  def export(self, name=None):
    """Returns tensors of all keys and values in the table.
    Args:
      name: A name for the operation (optional).
    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype)
    return exported_keys, exported_values
  def _serialize_to_tensors(self):
    """Implements checkpointing protocols for `Trackable`."""
    tensors = self.export()
    return {"-keys": tensors[0], "-values": tensors[1]}
  def _restore_from_tensors(self, restored_tensors):
    """Implements checkpointing protocols for `Trackable`."""
    with ops.name_scope("%s_table_restore" % self._name):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.resource_handle,
            restored_tensors["-keys"],
            restored_tensors["-values"])
    # This class is needed for `MutableHashTable(checkpoint=True)`.
  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for DenseHashTable."""
    def __init__(self, table, name, table_name=None):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      self.table_name = table_name or name
      # pylint: disable=protected-access
      super(MutableHashTable._Saveable, self).__init__(table, specs, name)
    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope("%s_table_restore" % self.table_name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])
