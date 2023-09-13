@tf_export("data.experimental.ignore_errors")
@deprecation.deprecated(None, "Use `tf.data.Dataset.ignore_errors` instead.")
def ignore_errors(log_warning=False):
  """Creates a `Dataset` from another `Dataset` and silently ignores any errors.
  Use this transformation to produce a dataset that contains the same elements
  as the input, but silently drops any elements that caused an error. For
  example:
  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])
  # Computing `tf.debugging.check_numerics(1. / 0.)` will raise an
  InvalidArgumentError.
  dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))
  # Using `ignore_errors()` will drop the element that causes an error.
  dataset =
      dataset.apply(tf.data.experimental.ignore_errors())  # ==> {1., 0.5, 0.2}
  ```
  Args:
     log_warning: (Optional.) A 'tf.bool' scalar indicating whether ignored
      errors should be logged to stderr. Defaults to 'False'.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.ignore_errors(log_warning)
  return _apply_fn
