@tf_export("data.experimental.at", v1=[])
def at(dataset, index):
  """Returns the element at a specific index in a datasest.
  Currently, random access is supported for the following tf.data operations:
     - `tf.data.Dataset.from_tensor_slices`,
     - `tf.data.Dataset.from_tensors`,
     - `tf.data.Dataset.shuffle`,
     - `tf.data.Dataset.batch`,
     - `tf.data.Dataset.shard`,
     - `tf.data.Dataset.map`,
     - `tf.data.Dataset.range`,
     - `tf.data.Dataset.zip`,
     - `tf.data.Dataset.skip`,
     - `tf.data.Dataset.repeat`,
     - `tf.data.Dataset.list_files`,
     - `tf.data.Dataset.SSTableDataset`,
     - `tf.data.Dataset.concatenate`,
     - `tf.data.Dataset.enumerate`,
     - `tf.data.Dataset.parallel_map`,
     - `tf.data.Dataset.prefetch`,
     - `tf.data.Dataset.take`,
     - `tf.data.Dataset.cache` (in-memory only)
     Users can use the cache operation to enable random access for any dataset,
     even one comprised of transformations which are not on this list.
     E.g., to get the third element of a TFDS dataset:
       ```python
       ds = tfds.load("mnist", split="train").cache()
       elem = tf.data.Dataset.experimental.at(ds, 3)
       ```
  Args:
    dataset: A `tf.data.Dataset` to determine whether it supports random access.
    index: The index at which to fetch the element.
  Returns:
      A (nested) structure of values matching `tf.data.Dataset.element_spec`.
   Raises:
     UnimplementedError: If random access is not yet supported for a dataset.
  """
  # pylint: disable=protected-access
  return structure.from_tensor_list(
      dataset.element_spec,
      gen_experimental_dataset_ops.get_element_at_index(
          dataset._variant_tensor,
          index,
          output_types=structure.get_flat_tensor_types(dataset.element_spec),
          output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec)))
