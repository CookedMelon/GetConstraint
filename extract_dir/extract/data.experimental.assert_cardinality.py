@tf_export("data.experimental.assert_cardinality")
def assert_cardinality(expected_cardinality):
  """Asserts the cardinality of the input dataset.
  NOTE: The following assumes that "examples.tfrecord" contains 42 records.
  >>> dataset = tf.data.TFRecordDataset("examples.tfrecord")
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())
  True
  >>> dataset = dataset.apply(tf.data.experimental.assert_cardinality(42))
  >>> print(tf.data.experimental.cardinality(dataset).numpy())
  42
  Args:
    expected_cardinality: The expected cardinality of the input dataset.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  Raises:
    FailedPreconditionError: The assertion is checked at runtime (when iterating
      the dataset) and an error is raised if the actual and expected cardinality
      differ.
  """
  def _apply_fn(dataset):
    return _AssertCardinalityDataset(dataset, expected_cardinality)
  return _apply_fn
