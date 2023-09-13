@tf_export("data.experimental.shuffle_and_repeat")
def shuffle_and_repeat(buffer_size, count=None, seed=None):
  """Shuffles and repeats a Dataset, reshuffling with each repetition.
  >>> d = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> d = d.apply(tf.data.experimental.shuffle_and_repeat(2, count=2))
  >>> [elem.numpy() for elem in d] # doctest: +SKIP
  [2, 3, 1, 1, 3, 2]
  ```python
  dataset.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size, count, seed))
  ```
  produces the same output as
  ```python
  dataset.shuffle(
    buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)
  ```
  In each repetition, this dataset fills a buffer with `buffer_size` elements,
  then randomly samples elements from this buffer, replacing the selected
  elements with new elements. For perfect shuffling, set the buffer size equal
  to the full size of the dataset.
  For instance, if your dataset contains 10,000 elements but `buffer_size` is
  set to 1,000, then `shuffle` will initially select a random element from
  only the first 1,000 elements in the buffer. Once an element is selected,
  its space in the buffer is replaced by the next (i.e. 1,001-st) element,
  maintaining the 1,000 element buffer.
  Args:
    buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
      number elements that will be buffered when prefetching.
    count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the number
      of times the dataset should be repeated. The default behavior (if `count`
      is `None` or `-1`) is for the dataset be repeated indefinitely.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):  # pylint: disable=missing-docstring
    return _ShuffleAndRepeatDataset(dataset, buffer_size, count, seed)
  return _apply_fn
