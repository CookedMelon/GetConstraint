@tf_export("data.experimental.table_from_dataset")
def table_from_dataset(dataset=None,
                       num_oov_buckets=0,
                       vocab_size=None,
                       default_value=None,
                       hasher_spec=lookup_ops.FastHashSpec,
                       key_dtype=dtypes.string,
                       name=None):
  """Returns a lookup table based on the given dataset.
  This operation constructs a lookup table based on the given dataset of pairs
  of (key, value).
  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.
  Sample Usages:
  >>> keys = tf.data.Dataset.range(100)
  >>> values = tf.data.Dataset.range(100).map(
  ...     lambda x: tf.strings.as_string(x * 2))
  >>> ds = tf.data.Dataset.zip((keys, values))
  >>> table = tf.data.experimental.table_from_dataset(
  ...                               ds, default_value='n/a', key_dtype=tf.int64)
  >>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
  array([b'0', b'2', b'4'], dtype=object)
  Args:
    dataset: A dataset containing (key, value) pairs.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).
  Returns:
    The lookup table based on the given dataset.
  Raises:
    ValueError: If
      * `dataset` does not contain pairs
      * The 2nd item in the `dataset` pairs has a dtype which is incompatible
        with `default_value`
      * `num_oov_buckets` is negative
      * `vocab_size` is not greater than zero
      * The `key_dtype` is not integer or string
  """
  elem_spec = dataset.element_spec
  _check_table_initializer_element_spec(elem_spec)
  if default_value is None:
    default_value = -1
    if not (elem_spec[1].dtype.is_integer or elem_spec[1].dtype.is_floating):
      raise ValueError("`default_value` must be specified when creating a "
                       "table from a dataset that produces values of type "
                       f"{elem_spec[1].dtype}.")
  if num_oov_buckets < 0:
    raise ValueError("`num_oov_buckets` must be greater than or equal to 0, "
                     f"got {num_oov_buckets}.")
  if (not isinstance(vocab_size, ops.Tensor) and vocab_size is not None and
      vocab_size < 1):
    raise ValueError(f"`vocab_size` must be greater than 0, got {vocab_size}.")
  if (not key_dtype.is_integer) and (dtypes.string != key_dtype.base_dtype):
    raise TypeError("`key_dtype` must be either an integer or string type, "
                    f"but got {key_dtype}")
  if vocab_size is not None:
    if isinstance(vocab_size, ops.Tensor):
      vocab_size = math_ops.cast(vocab_size, dtypes.int64)
    dataset = dataset.take(vocab_size)
    dataset = dataset.apply(assert_cardinality(vocab_size))
  with ops.name_scope(name, "string_to_index"):
    initializer = DatasetInitializer(dataset)
    with ops.name_scope(None, "hash_table"):
      table = lookup_ops.StaticHashTableV1(initializer, default_value)
      if num_oov_buckets:
        table = lookup_ops.IdTableWithHashBuckets(
            table,
            num_oov_buckets=num_oov_buckets,
            hasher_spec=hasher_spec,
            key_dtype=key_dtype)
      return table
