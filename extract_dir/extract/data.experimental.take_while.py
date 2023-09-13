@tf_export("data.experimental.take_while")
def take_while(predicate):
  """A transformation that stops dataset iteration based on a `predicate`.
  Args:
    predicate: A function that maps a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to a
      scalar `tf.bool` tensor.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.take_while(predicate=predicate)
  return _apply_fn
