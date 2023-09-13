@tf_export("data.experimental.unique")
def unique():
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.
  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:
  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])
  # Using `unique()` will drop the duplicate elements.
  dataset = dataset.apply(tf.data.experimental.unique())  # ==> { 1, 37, 2 }
  ```
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.unique()
  return _apply_fn
