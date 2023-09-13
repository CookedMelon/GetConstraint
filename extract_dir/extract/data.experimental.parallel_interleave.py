@tf_export("data.experimental.parallel_interleave")
def parallel_interleave(map_func,
                        cycle_length,
                        block_length=1,
                        sloppy=False,
                        buffer_output_elements=None,
                        prefetch_input_elements=None):
  """A parallel version of the `Dataset.interleave()` transformation.
  `parallel_interleave()` maps `map_func` across its input to produce nested
  datasets, and outputs their elements interleaved. Unlike
  `tf.data.Dataset.interleave`, it gets elements from `cycle_length` nested
  datasets in parallel, which increases the throughput, especially in the
  presence of stragglers. Furthermore, the `sloppy` argument can be used to
  improve performance, by relaxing the requirement that the outputs are produced
  in a deterministic order, and allowing the implementation to skip over nested
  datasets whose elements are not readily available when requested.
  Example usage:
  ```python
  # Preprocess 4 files concurrently.
  filenames = tf.data.Dataset.list_files("/path/to/data/train*.tfrecords")
  dataset = filenames.apply(
      tf.data.experimental.parallel_interleave(
          lambda filename: tf.data.TFRecordDataset(filename),
          cycle_length=4))
  ```
  WARNING: If `sloppy` is `True`, the order of produced elements is not
  deterministic.
  Args:
    map_func: A function mapping a nested structure of tensors to a `Dataset`.
    cycle_length: The number of input `Dataset`s to interleave from in parallel.
    block_length: The number of consecutive elements to pull from an input
      `Dataset` before advancing to the next input `Dataset`.
    sloppy: A boolean controlling whether determinism should be traded for
      performance by allowing elements to be produced out of order.  If `sloppy`
      is `None`, the `tf.data.Options.deterministic` dataset option (`True` by
      default) is used to decide whether to enforce a deterministic order.
    buffer_output_elements: The number of elements each iterator being
      interleaved should buffer (similar to the `.prefetch()` transformation for
      each interleaved iterator).
    prefetch_input_elements: The number of input elements to transform to
      iterators before they are needed for interleaving.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return readers.ParallelInterleaveDataset(dataset, map_func, cycle_length,
                                             block_length, sloppy,
                                             buffer_output_elements,
                                             prefetch_input_elements)
  return _apply_fn
