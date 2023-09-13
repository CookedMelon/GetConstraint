@tf_export("data.experimental.group_by_reducer")
def group_by_reducer(key_func, reducer):
  """A transformation that groups elements and performs a reduction.
  This transformation maps element of a dataset to a key using `key_func` and
  groups the elements by key. The `reducer` is used to process each group; its
  `init_func` is used to initialize state for each group when it is created, the
  `reduce_func` is used to update the state every time an element is mapped to
  the matching group, and the `finalize_func` is used to map the final state to
  an output value.
  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reducer: An instance of `Reducer`, which captures the reduction logic using
      the `init_func`, `reduce_func`, and `finalize_func` functions.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _GroupByReducerDataset(dataset, key_func, reducer)
  return _apply_fn
