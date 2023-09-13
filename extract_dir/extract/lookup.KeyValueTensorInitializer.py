@tf_export("lookup.KeyValueTensorInitializer")
class KeyValueTensorInitializer(TableInitializerBase):
  """Table initializers given `keys` and `values` tensors.
  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
  >>> table = tf.lookup.StaticHashTable(
  ...     init,
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)
  """
  def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
    """Constructs a table initializer object based on keys and values tensors.
    Args:
      keys: The tensor for the keys.
      values: The tensor for the values.
      key_dtype: The `keys` data type. Used when `keys` is a python array.
      value_dtype: The `values` data type. Used when `values` is a python array.
      name: A name for the operation (optional).
    """
    if (not context.executing_eagerly() and
        ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with ops.init_scope():
        self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
        self._values = ops.convert_to_tensor(
            values, dtype=value_dtype, name="values")
    else:
      self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
      self._values = ops.convert_to_tensor(
          values, dtype=value_dtype, name="values")
    self._name = name if name is not None else "key_value_init"
    if context.executing_eagerly():
      # Ensure a unique name when eager execution is enabled to avoid spurious
      # sharing issues.
      # TODO(rohanj): Use context.anonymous_name() instead.
      self._name += str(ops.uid())
    super(KeyValueTensorInitializer, self).__init__(self._keys.dtype,
                                                    self._values.dtype)
  def initialize(self, table):
    """Initializes the given `table` with `keys` and `values` tensors.
    Args:
      table: The table to initialize.
    Returns:
      The operation that initializes the table.
    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    check_table_dtypes(table, self._keys.dtype, self._values.dtype)
    with ops.name_scope(
        self._name, values=(table.resource_handle, self._keys, self._values)):
      init_op = gen_lookup_ops.lookup_table_import_v2(table.resource_handle,
                                                      self._keys, self._values)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op
