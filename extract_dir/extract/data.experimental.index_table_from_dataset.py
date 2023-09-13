@tf_export("data.experimental.index_table_from_dataset")
def index_table_from_dataset(dataset=None,
                             num_oov_buckets=0,
                             vocab_size=None,
                             default_value=-1,
                             hasher_spec=lookup_ops.FastHashSpec,
                             key_dtype=dtypes.string,
                             name=None):
  """Returns an index lookup table based on the given dataset.
  This operation constructs a lookup table based on the given dataset of keys.
  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.
  Sample Usages:
  >>> ds = tf.data.Dataset.range(100).map(lambda x: tf.strings.as_string(x * 2))
  >>> table = tf.data.experimental.index_table_from_dataset(
  ...                                     ds, key_dtype=dtypes.int64)
  >>> table.lookup(tf.constant(['0', '2', '4'], dtype=tf.string)).numpy()
  array([0, 1, 2])
  Args:
    dataset: A dataset of keys.
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
      * `num_oov_buckets` is negative
      * `vocab_size` is not greater than zero
      * The `key_dtype` is not integer or string
  """
  return table_from_dataset(dataset.enumerate().map(lambda v, k: (k, v)),
                            num_oov_buckets, vocab_size, default_value,
                            hasher_spec, key_dtype, name)
