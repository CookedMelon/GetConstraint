@tf_export("data.experimental.cardinality")
def cardinality(dataset):
  """Returns the cardinality of `dataset`, if known.
  The operation returns the cardinality of `dataset`. The operation may return
  `tf.data.experimental.INFINITE_CARDINALITY` if `dataset` contains an infinite
  number of elements or `tf.data.experimental.UNKNOWN_CARDINALITY` if the
  analysis fails to determine the number of elements in `dataset` (e.g. when the
  dataset source is a file).
  >>> dataset = tf.data.Dataset.range(42)
  >>> print(tf.data.experimental.cardinality(dataset).numpy())
  42
  >>> dataset = dataset.repeat()
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.INFINITE_CARDINALITY).numpy())
  True
  >>> dataset = dataset.filter(lambda x: True)
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())
  True
  Args:
    dataset: A `tf.data.Dataset` for which to determine cardinality.
  Returns:
    A scalar `tf.int64` `Tensor` representing the cardinality of `dataset`. If
    the cardinality is infinite or unknown, the operation returns the named
    constant `INFINITE_CARDINALITY` and `UNKNOWN_CARDINALITY` respectively.
  """
  return gen_dataset_ops.dataset_cardinality(dataset._variant_tensor)  # pylint: disable=protected-access
