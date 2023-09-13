@tf_export("data.experimental.from_list")
def from_list(elements, name=None):
  """Creates a `Dataset` comprising the given list of elements.
  The returned dataset will produce the items in the list one by one. The
  functionality is identical to `Dataset.from_tensor_slices` when elements are
  scalars, but different when elements have structure. Consider the following
  example.
  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])
  >>> list(dataset.as_numpy_iterator())
  [(1, b'a'), (2, b'b'), (3, b'c')]
  To get the same output with `from_tensor_slices`, the data needs to be
  reorganized:
  >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'c']))
  >>> list(dataset.as_numpy_iterator())
  [(1, b'a'), (2, b'b'), (3, b'c')]
  Unlike `from_tensor_slices`, `from_list` supports non-rectangular input:
  >>> dataset = tf.data.experimental.from_list([[1], [2, 3]])
  >>> list(dataset.as_numpy_iterator())
  [array([1], dtype=int32), array([2, 3], dtype=int32)]
  Achieving the same with `from_tensor_slices` requires the use of ragged
  tensors.
  `from_list` can be more performant than `from_tensor_slices` in some cases,
  since it avoids the need for data slicing each epoch. However, it can also be
  less performant, because data is stored as many small tensors rather than a
  few large tensors as in `from_tensor_slices`. The general guidance is to
  prefer `from_list` from a performance perspective when the number of elements
  is small (less than 1000).
  Args:
    elements: A list of elements whose components have the same nested
      structure.
    name: (Optional.) A name for the tf.data operation.
  Returns:
    Dataset: A `Dataset` of the `elements`.
  """
  return _ListDataset(elements, name)
