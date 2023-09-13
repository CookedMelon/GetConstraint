@tf_export("lookup.StaticVocabularyTable", v1=[])
class StaticVocabularyTable(LookupInterface):
  r"""String to Id table that assigns out-of-vocabulary keys to hash buckets.
  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:
  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(['emerson', 'lake', 'palmer']),
  ...     values=tf.constant([0, 1, 2], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...    init,
  ...    num_oov_buckets=5)
  The `Vocabulary` object will performs the following mapping:
  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where `bucket_id` will be between `3` and
  `3 + num_oov_buckets - 1 = 7`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`
  If input_tensor is:
  >>> input_tensor = tf.constant(["emerson", "lake", "palmer",
  ...                             "king", "crimson"])
  >>> table[input_tensor].numpy()
  array([0, 1, 2, 6, 7])
  If `initializer` is None, only out-of-vocabulary buckets are used.
  Example usage:
  >>> num_oov_buckets = 3
  >>> vocab = ["emerson", "lake", "palmer", "crimnson"]
  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> f.write('\n'.join(vocab).encode('utf-8'))
  >>> f.close()
  >>> init = tf.lookup.TextFileInitializer(
  ...     f.name,
  ...     key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...     value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
  >>> table.lookup(tf.constant(["palmer", "crimnson" , "king",
  ...                           "tarkus", "black", "moon"])).numpy()
  array([2, 3, 5, 6, 6, 4])
  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.
  Note that the out-of-vocabulary bucket IDs always range from the table `size`
  up to `size + num_oov_buckets - 1` regardless of the table values, which could
  cause unexpected collisions:
  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(["emerson", "lake", "palmer"]),
  ...     values=tf.constant([1, 2, 3], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...     init,
  ...     num_oov_buckets=1)
  >>> input_tensor = tf.constant(["emerson", "lake", "palmer", "king"])
  >>> table[input_tensor].numpy()
  array([1, 2, 3, 3])
  """
  def __init__(self,
               initializer,
               num_oov_buckets,
               lookup_key_dtype=None,
               name=None,
               experimental_is_anonymous=False):
    """Construct a `StaticVocabularyTable` object.
    Args:
      initializer: A `TableInitializerBase` object that contains the data used
        to initialize the table. If None, then we only use out-of-vocab buckets.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys. Must
        be greater than zero. If out-of-vocab buckets are not required, use
        `StaticHashTable` instead.
      lookup_key_dtype: Data type of keys passed to `lookup`. Defaults to
        `initializer.key_dtype` if `initializer` is specified, otherwise
        `tf.string`. Must be string or integer, and must be castable to
        `initializer.key_dtype`.
      name: A name for the operation (optional).
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.
    Raises:
      ValueError: when `num_oov_buckets` is not positive.
      TypeError: when lookup_key_dtype or initializer.key_dtype are not
        integer or string. Also when initializer.value_dtype != int64.
    """
    if num_oov_buckets <= 0:
      raise ValueError("`num_oov_buckets` must be > 0; use StaticHashTable.")
    # If a name ends with a '/' it is a "name scope", remove all trailing '/'
    # characters to use as table name.
    if name:
      name = name.rstrip("/")
    if initializer:
      if lookup_key_dtype is None:
        lookup_key_dtype = initializer.key_dtype
      supported_table_key_dtypes = (dtypes.int64, dtypes.string)
      if initializer.key_dtype not in supported_table_key_dtypes:
        raise TypeError("Invalid `key_dtype`, expected one of %s, but got %s." %
                        (supported_table_key_dtypes, initializer.key_dtype))
      if initializer.key_dtype.is_integer != lookup_key_dtype.is_integer:
        raise TypeError(
            "Invalid `key_dtype`, expected %s but got %s." %
            ("integer" if lookup_key_dtype.is_integer else "non-integer",
             initializer.key_dtype))
      if initializer.value_dtype != dtypes.int64:
        raise TypeError("Invalid `value_dtype`, expected %s but got %s." %
                        (dtypes.int64, initializer.value_dtype))
      if isinstance(initializer, trackable_base.Trackable):
        self._initializer = self._track_trackable(initializer, "_initializer")
      self._table = HashTable(
          initializer,
          default_value=-1,
          experimental_is_anonymous=experimental_is_anonymous)
      name = name or self._table.name
    else:
      lookup_key_dtype = dtypes.string
      self._table = None
      name = name or "hash_bucket"
    if (not lookup_key_dtype.is_integer) and (dtypes.string !=
                                              lookup_key_dtype):
      raise TypeError("Invalid `key_dtype`, expected integer or string, got "
                      f"{lookup_key_dtype}")
    self._num_oov_buckets = num_oov_buckets
    self._table_name = None
    if name is not None:
      self._table_name = name.split("/")[-1]
    super(StaticVocabularyTable, self).__init__(lookup_key_dtype, dtypes.int64)
  def _create_resource(self):
    if self._table is not None:
      return self._table._create_resource()  # pylint: disable=protected-access
    return None
  def _initialize(self):
    if self._table is not None:
      return self._table._initialize()  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()
  @property
  def resource_handle(self):
    if self._table is not None:
      return self._table.resource_handle
    return None
  @property
  def name(self):
    return self._table_name
  def size(self, name=None):
    """Compute the number of elements in this table."""
    with ops.name_scope(name, "%s_Size" % self.name):
      if self._table:
        tsize = self._table.size()
      else:
        tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
      return tsize + self._num_oov_buckets
  def lookup(self, keys, name=None):
    """Looks up `keys` in the table, outputs the corresponding values.
    It assigns out-of-vocabulary keys to buckets based in their hashes.
    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.
    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.
    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")
    values = keys
    if isinstance(keys,
                  (sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor)):
      values = keys.values
    if self._table and (self._table.key_dtype.base_dtype == dtypes.int64):
      values = math_ops.cast(values, dtypes.int64)
    # TODO(yleon): Consider moving this functionality to its own kernel.
    with ops.name_scope(name, "%s_Lookup" % self.name):
      buckets = string_ops.string_to_hash_bucket_fast(
          _as_string(values),
          num_buckets=self._num_oov_buckets,
          name="hash_bucket")
      if self._table:
        ids = self._table.lookup(values)
        buckets = math_ops.add(buckets, self._table.size())
        is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
        ids = array_ops.where_v2(is_id_non_default, ids, buckets)
      else:
        ids = buckets
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
    elif isinstance(keys, ragged_tensor.RaggedTensor):
      return keys.with_values(ids)
    return ids
