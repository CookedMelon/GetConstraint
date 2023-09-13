@tf_export("data.experimental.get_structure")
def get_structure(dataset_or_iterator):
  """Returns the type signature for elements of the input dataset / iterator.
  For example, to get the structure of a `tf.data.Dataset`:
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> tf.data.experimental.get_structure(dataset)
  TensorSpec(shape=(), dtype=tf.int32, name=None)
  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])
  >>> tf.data.experimental.get_structure(dataset)
  (TensorSpec(shape=(), dtype=tf.int32, name=None),
   TensorSpec(shape=(), dtype=tf.string, name=None))
  To get the structure of an `tf.data.Iterator`:
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> tf.data.experimental.get_structure(iter(dataset))
  TensorSpec(shape=(), dtype=tf.int32, name=None)
  Args:
    dataset_or_iterator: A `tf.data.Dataset` or an `tf.data.Iterator`.
  Returns:
    A (nested) structure of `tf.TypeSpec` objects matching the structure of an
    element of `dataset_or_iterator` and specifying the type of individual
    components.
  Raises:
    TypeError: If input is not a `tf.data.Dataset` or an `tf.data.Iterator`
      object.
  """
  try:
    return dataset_or_iterator.element_spec  # pylint: disable=protected-access
  except AttributeError:
    raise TypeError(f"Invalid `dataset_or_iterator`. `dataset_or_iterator` "
                    f"must be a `tf.data.Dataset` or tf.data.Iterator object, "
                    f"but got {type(dataset_or_iterator)}.")
