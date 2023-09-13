@tf_export("lookup.experimental.DenseHashTable")
@saveable_compat.legacy_saveable_name("table")
class DenseHashTable(LookupInterface):
  """A mutable hash table with faster lookups and higher memory usage.
  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.
  Compared to `MutableHashTable`, `DenseHashTable` offers generally faster
  `insert`, `remove` and `lookup` operations, in exchange for a higher overall
  memory footprint.
  It uses "open addressing" with quadratic reprobing to resolve collisions. This
  requires specifying two keys in the key space, `empty_key` and `deleted_key`,
  that can never inserted into the table.
  Unlike `MutableHashTable`, `DenseHashTable` does not require additional memory
  for temporary tensors created during checkpointing and restore operations.
  Example usage:
  >>> table = tf.lookup.experimental.DenseHashTable(
  ...     key_dtype=tf.string,
  ...     value_dtype=tf.int64,
  ...     default_value=-1,
  ...     empty_key='',
  ...     deleted_key='$')
  >>> keys = tf.constant(['a', 'b', 'c'])
  >>> values = tf.constant([0, 1, 2], dtype=tf.int64)
  >>> table.insert(keys, values)
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(tf.constant(['a', 'b', 'c','d'])).numpy()
  array([ 0,  1, -1, -1])
  """
  # TODO(andreasst): consider extracting common code with MutableHashTable into
  # a common superclass.
  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               empty_key,
               deleted_key,
               initial_num_buckets=None,
               name="MutableDenseHashTable",
               checkpoint=True,
               experimental_is_anonymous=False):
    """Creates an empty `DenseHashTable` object.
    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.
    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert, remove or lookup operations.
      deleted_key: the key to use to represent deleted buckets internally. Must
        not be used in insert, remove or lookup operations and be different from
        the empty_key.
      initial_num_buckets: the initial number of buckets (optional,
        default to 2^17=131072). Note that the default value is
        relatively large (~1MB), so if you are going to create many
        tables (likely the case when `experimental_is_anonymous` is
        `True`), you should set `initial_num_buckets` to a smaller
        value to reduce memory usage.
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
      A `DenseHashTable` object.
    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype, name="default_value")
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    # TODO(b/201578996): Pick a good default for initial_num_buckets
    #   other than 2^17.
    self._initial_num_buckets = initial_num_buckets
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._name = name
    self._empty_key = empty_key
    self._deleted_key = deleted_key
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = None
      if context.executing_eagerly():
        # TODO(allenl): This will leak memory due to kernel caching by
        # the shared_name attribute value (but is better than the
        # alternative of sharing everything by default when executing
        # eagerly; hopefully creating tables in a loop is uncommon).
        self._shared_name = "table_%d" % (ops.uid(),)
    super(DenseHashTable, self).__init__(key_dtype, value_dtype)
    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = DenseHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
  def _create_resource(self):
    empty_key = ops.convert_to_tensor(
        self._empty_key, dtype=self._key_dtype, name="empty_key")
    deleted_key = ops.convert_to_tensor(
        self._deleted_key, dtype=self._key_dtype, name="deleted_key")
    if self._is_anonymous:
      table_ref = gen_lookup_ops.anonymous_mutable_dense_hash_table(
          empty_key=empty_key,
          deleted_key=deleted_key,
          value_dtype=self._value_dtype,
          value_shape=self._value_shape,
          initial_num_buckets=self._initial_num_buckets,
          name=self._name)
    else:
      # The table must be shared if checkpointing is requested for multi-worker
      # training to work correctly. Use the node name if no shared_name has been
      # explicitly specified.
      use_node_name_sharing = self._checkpoint and self._shared_name is None
      table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(
          empty_key=empty_key,
          deleted_key=deleted_key,
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          value_dtype=self._value_dtype,
          value_shape=self._value_shape,
          initial_num_buckets=self._initial_num_buckets,
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
  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.
    The `default_value` is used for keys not present in the table.
    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).
    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.
    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        [self.resource_handle, keys]):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys,
                                                     self._default_value)
    return values
  def insert_or_assign(self, keys, values, name=None):
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
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      values = ops.convert_to_tensor(
          values, dtype=self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
      return op
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
    return self.insert_or_assign(keys, values, name)
  def erase(self, keys, name=None):
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
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))
    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)
    return op
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
    return self.erase(keys, name)
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
    """Implements checkpointing interface in `Trackable`."""
    tensors = self.export()
    return {"-keys": tensors[0], "-values": tensors[1]}
  def _restore_from_tensors(self, restored_tensors):
    """Implements checkpointing interface in `Trackable`."""
    with ops.name_scope("%s_table_restore" % self._name):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.resource_handle,
            restored_tensors["-keys"],
            restored_tensors["-values"])
  # This class is needed for `DenseHashTable(checkpoint=True)`.
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
      super(DenseHashTable._Saveable, self).__init__(table, specs, name)
    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope("%s_table_restore" % self.table_name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])
