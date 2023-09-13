@tf_export("data.experimental.choose_from_datasets", v1=[])
def choose_from_datasets_v2(datasets,
                            choice_dataset,
                            stop_on_empty_dataset=False):
  """Creates a dataset that deterministically chooses elements from `datasets`.
  For example, given the following datasets:
  ```python
  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
              tf.data.Dataset.from_tensors("bar").repeat(),
              tf.data.Dataset.from_tensors("baz").repeat()]
  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
  choice_dataset = tf.data.Dataset.range(3).repeat(3)
  result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
  ```
  The elements of `result` will be:
  ```
  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
  ```
  Args:
    datasets: A non-empty list of `tf.data.Dataset` objects with compatible
      structure.
    choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between `0`
      and `len(datasets) - 1`.
    stop_on_empty_dataset: If `True`, selection stops if it encounters an empty
      dataset. If `False`, it skips empty datasets. It is recommended to set it
      to `True`. Otherwise, the selected elements start off as the user intends,
      but may change as input datasets become empty. This can be difficult to
      detect since the dataset starts off looking correct. Default to `False`
      for backward compatibility.
  Returns:
    A dataset that interleaves elements from `datasets` according to the values
    of `choice_dataset`.
  Raises:
    TypeError: If `datasets` or `choice_dataset` has the wrong type.
    ValueError: If `datasets` is empty.
  """
  return dataset_ops.Dataset.choose_from_datasets(
      datasets=datasets,
      choice_dataset=choice_dataset,
      stop_on_empty_dataset=stop_on_empty_dataset)
