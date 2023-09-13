@tf_export("data.experimental.unbatch")
def unbatch():
  """Splits elements of a dataset into multiple elements on the batch dimension.
  For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
  where `B` may vary for each input element, then for each element in the
  dataset, the unbatched dataset will contain `B` consecutive elements
  of shape `[a0, a1, ...]`.
  ```python
  # NOTE: The following example uses `{ ... }` to represent the contents
  # of a dataset.
  a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }
  a.unbatch() == {
      'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
  ```
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.unbatch()
  return _apply_fn
