@tf_export("data.experimental.Counter", v1=[])
@deprecation.deprecated(None, "Use `tf.data.Dataset.counter(...)` instead.")
def CounterV2(start=0, step=1, dtype=dtypes.int64):
  """Creates a `Dataset` that counts from `start` in steps of size `step`.
  Unlike `tf.data.Dataset.range` which will stop at some ending number,
  `Counter` will produce elements indefinitely.
  >>> dataset = tf.data.experimental.Counter().take(5)
  >>> list(dataset.as_numpy_iterator())
  [0, 1, 2, 3, 4]
  >>> dataset.element_spec
  TensorSpec(shape=(), dtype=tf.int64, name=None)
  >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
  >>> dataset.element_spec
  TensorSpec(shape=(), dtype=tf.int32, name=None)
  >>> dataset = tf.data.experimental.Counter(start=2).take(5)
  >>> list(dataset.as_numpy_iterator())
  [2, 3, 4, 5, 6]
  >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
  >>> list(dataset.as_numpy_iterator())
  [2, 7, 12, 17, 22]
  >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
  >>> list(dataset.as_numpy_iterator())
  [10, 9, 8, 7, 6]
  Args:
    start: (Optional.) The starting value for the counter. Defaults to 0.
    step: (Optional.) The step size for the counter. Defaults to 1.
    dtype: (Optional.) The data type for counter elements. Defaults to
      `tf.int64`.
  Returns:
    A `Dataset` of scalar `dtype` elements.
  """
  return dataset_ops.Dataset.counter(start, step, dtype)
