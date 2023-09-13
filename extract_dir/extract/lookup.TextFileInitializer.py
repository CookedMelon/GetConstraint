@tf_export("lookup.TextFileInitializer")
class TextFileInitializer(TableInitializerBase):
  r"""Table initializers from a text file.
  This initializer assigns one entry in the table for each line in the file.
  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.
  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  For example if we have a file with the following content:
  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> content='\n'.join(["emerson 10", "lake 20", "palmer 30",])
  >>> f.file.write(content.encode('utf-8'))
  >>> f.file.close()
  The following snippet initializes a table with the first column as keys and
  second column as values:
  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`
  >>> init= tf.lookup.TextFileInitializer(
  ...    filename=f.name,
  ...    key_dtype=tf.string, key_index=0,
  ...    value_dtype=tf.int64, value_index=1,
  ...    delimiter=" ")
  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)
  >>> table.lookup(tf.constant(['palmer','lake','tarkus'])).numpy()
  Similarly to initialize the whole line as keys and the line number as values.
  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`
  >>> init = tf.lookup.TextFileInitializer(
  ...   filename=f.name,
  ...   key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...   value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticHashTable(init, -1)
  >>> table.lookup(tf.constant('palmer 30')).numpy()
  2
  """
  def __init__(self,
               filename,
               key_dtype,
               key_index,
               value_dtype,
               value_index,
               vocab_size=None,
               delimiter="\t",
               name=None,
               value_index_offset=0):
    """Constructs a table initializer object to populate from a text file.
    It generates one key-value pair per line. The type of table key and
    value are specified by `key_dtype` and `value_dtype`, respectively.
    Similarly the content of the key and value are specified by the key_index
    and value_index.
    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string or int64.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.
    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_dtype: The `key` data type.
      key_index: the index that represents information of a line to get the
        table 'key' values from.
      value_dtype: The `value` data type.
      value_index: the index that represents information of a line to get the
        table 'value' values from.'
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: A name for the operation (optional).
      value_index_offset: A number to add to all indices extracted from the file
        This is useful for cases where a user would like to reserve one or more
        low index values for control characters. For instance, if you would
        like to ensure that no vocabulary item is mapped to index 0 (so you can
        reserve 0 for a masking value), you can set value_index_offset to 1;
        this will mean that the first vocabulary element is mapped to 1
        instead of 0.
    Raises:
      ValueError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    if not isinstance(filename, ops.Tensor) and not filename:
      raise ValueError("`filename` argument required for tf.lookup.TextFileInitializer")
    self._filename_arg = filename
    key_dtype = dtypes.as_dtype(key_dtype)
    value_dtype = dtypes.as_dtype(value_dtype)
    if key_index < -2:
      raise ValueError(f"`key_index` should be >= -2, received: {key_index}.")
    if key_index == TextFileIndex.LINE_NUMBER and key_dtype != dtypes.int64:
      raise ValueError("`key_dtype` must be int64 if `key_index` is "
                       f"{TextFileIndex.LINE_NUMBER}, received: {key_dtype}")
    if ((key_index == TextFileIndex.WHOLE_LINE) and
        (not key_dtype.is_integer) and (key_dtype != dtypes.string)):
      raise ValueError(
          "`key_dtype` should be either integer or string for `key_index` "
          f"{TextFileIndex.WHOLE_LINE}, received: {key_dtype}")
    if value_index < -2:
      raise ValueError("`value_index` should be >= -2, received: "
                       f"{value_index}")
    if value_index == TextFileIndex.LINE_NUMBER and value_dtype != dtypes.int64:
      raise ValueError("`value_dtype` must be int64 for `value_index` "
                       f"{TextFileIndex.LINE_NUMBER}, received: {value_dtype}")
    if ((value_index == TextFileIndex.WHOLE_LINE) and
        (not value_dtype.is_integer) and (value_dtype != dtypes.string)):
      raise ValueError(
          "`value_dtype` should be either integer or string for `value_index` "
          f"{TextFileIndex.WHOLE_LINE}, received: {value_dtype}")
    if (vocab_size is not None) and (vocab_size <= 0):
      raise ValueError(f"`vocab_size` should be > 0, received: {vocab_size}")
    self._key_index = key_index
    self._value_index = value_index
    self._vocab_size = vocab_size
    self._delimiter = delimiter
    self._name = name
    self._filename = self._track_trackable(
        asset.Asset(filename), "_filename")
    self._offset = value_index_offset
    super(TextFileInitializer, self).__init__(key_dtype, value_dtype)
  def initialize(self, table):
    """Initializes the table from a text file.
    Args:
      table: The table to be initialized.
    Returns:
      The operation that initializes the table.
    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    check_table_dtypes(table, self.key_dtype, self.value_dtype)
    with ops.name_scope(self._name, "text_file_init", (table.resource_handle,)):
      filename = ops.convert_to_tensor(
          self._filename, dtypes.string, name="asset_filepath")
      init_op = gen_lookup_ops.initialize_table_from_text_file_v2(
          table.resource_handle, filename, self._key_index, self._value_index,
          -1 if self._vocab_size is None else self._vocab_size, self._delimiter,
          self._offset)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    # If the filename tensor is anything other than a string constant (e.g.,
    # if it is a placeholder) then it does not make sense to track it as an
    # asset.
    if not context.executing_eagerly() and constant_op.is_constant(filename):
      ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
    return init_op
  @property
  def _shared_name(self):
    if self._vocab_size:
      # Keep the shared_name:
      # <table_type>_<filename>_<vocab_size>_<key_index>_<value_index>_<offset>
      if self._offset:
        shared_name = "hash_table_%s_%d_%s_%s_%s" % (
            self._filename_arg, self._vocab_size, self._key_index,
            self._value_index, self._offset)
      else:
        shared_name = "hash_table_%s_%d_%s_%s" % (
            self._filename_arg, self._vocab_size, self._key_index,
            self._value_index)
    else:
      # Keep the shared_name
      # <table_type>_<filename>_<key_index>_<value_index>_<offset>
      if self._offset:
        shared_name = "hash_table_%s_%s_%s_%s" % (
            self._filename_arg, self._key_index, self._value_index,
            self._offset)
      else:
        shared_name = "hash_table_%s_%s_%s" % (
            self._filename_arg, self._key_index, self._value_index)
    return shared_name
