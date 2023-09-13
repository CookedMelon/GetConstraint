@tf_export("data.experimental.get_next_as_optional")
def get_next_as_optional(iterator):
  """Returns a `tf.experimental.Optional` with the next element of the iterator.
  If the iterator has reached the end of the sequence, the returned
  `tf.experimental.Optional` will have no value.
  Args:
    iterator: A `tf.data.Iterator`.
  Returns:
    A `tf.experimental.Optional` object which either contains the next element
    of the iterator (if it exists) or no value.
  """
  return iterator.get_next_as_optional()
