@tf_export("data.experimental.make_batched_features_dataset", v1=[])
def make_batched_features_dataset_v2(file_pattern,
                                     batch_size,
                                     features,
                                     reader=None,
                                     label_key=None,
                                     reader_args=None,
                                     num_epochs=None,
                                     shuffle=True,
                                     shuffle_buffer_size=10000,
                                     shuffle_seed=None,
                                     prefetch_buffer_size=None,
                                     reader_num_threads=None,
                                     parser_num_threads=None,
                                     sloppy_ordering=False,
                                     drop_final_batch=False):
  """Returns a `Dataset` of feature dictionaries from `Example` protos.
  If label_key argument is provided, returns a `Dataset` of tuple
  comprising of feature dictionaries and label.
  Example:
  ```
  serialized_examples = [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "code", "art" ] } } }
    },
    features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "sports" ] } } }
    }
  ]
  ```
  We can use arguments:
  ```
  features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
    "kws": VarLenFeature(dtype=tf.string),
  }
  ```
  And the expected output is:
  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
    "kws": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["code", "art", "sports"]
      dense_shape=[2, 2]),
  }
  ```
  Args:
    file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.io.parse_example`.
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    label_key: (Optional) A string corresponding to the key labels are stored in
      `tf.Examples`. If provided, it must be one of the `features` key,
      otherwise results in `ValueError`.
    reader_args: Additional arguments to pass to the reader class.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. Defaults to `None`.
    shuffle: A boolean, indicates whether the input should be shuffled. Defaults
      to `True`.
    shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
      ensures better shuffling but would increase memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: Number of feature batches to prefetch in order to
      improve performance. Recommended value is the number of batches consumed
      per training step. Defaults to auto-tune.
    reader_num_threads: Number of threads used to read `Example` records. If >1,
      the results will be interleaved. Defaults to `1`.
    parser_num_threads: Number of threads to use for parsing `Example` tensors
      into a dictionary of `Feature` tensors. Defaults to `2`.
    sloppy_ordering: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    drop_final_batch: If `True`, and the batch size does not evenly divide the
      input dataset size, the final smaller batch will be dropped. Defaults to
      `False`.
  Returns:
    A dataset of `dict` elements, (or a tuple of `dict` elements and label).
    Each `dict` maps feature keys to `Tensor` or `SparseTensor` objects.
  Raises:
    TypeError: If `reader` is of the wrong type.
    ValueError: If `label_key` is not one of the `features` keys.
  """
  if reader is None:
    reader = core_readers.TFRecordDataset
  if reader_num_threads is None:
    reader_num_threads = 1
  if parser_num_threads is None:
    parser_num_threads = 2
  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE
  # Create dataset of all matching filenames
  dataset = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)
  if isinstance(reader, type) and issubclass(reader, io_ops.ReaderBase):
    raise TypeError("The `reader` argument must return a `Dataset` object. "
                    "`tf.ReaderBase` subclasses are not supported. For "
                    "example, pass `tf.data.TFRecordDataset` instead of "
                    "`tf.TFRecordReader`.")
  # Read `Example` records from files as tensor objects.
  if reader_args is None:
    reader_args = []
  if reader_num_threads == dataset_ops.AUTOTUNE:
    dataset = dataset.interleave(
        lambda filename: reader(filename, *reader_args),
        num_parallel_calls=reader_num_threads)
    options = options_lib.Options()
    options.deterministic = not sloppy_ordering
    dataset = dataset.with_options(options)
  else:
    # Read files sequentially (if reader_num_threads=1) or in parallel
    def apply_fn(dataset):
      return core_readers.ParallelInterleaveDataset(
          dataset,
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          block_length=1,
          sloppy=sloppy_ordering,
          buffer_output_elements=None,
          prefetch_input_elements=None)
    dataset = dataset.apply(apply_fn)
  # Extract values if the `Example` tensors are stored as key-value tuples.
  if dataset_ops.get_legacy_output_types(dataset) == (
      dtypes.string, dtypes.string):
    dataset = map_op._MapDataset(  # pylint: disable=protected-access
        dataset, lambda _, v: v, use_inter_op_parallelism=False)
  # Apply dataset repeat and shuffle transformations.
  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)
  # NOTE(mrry): We set `drop_remainder=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)
  # Parse `Example` tensors to a dictionary of `Feature` tensors.
  dataset = dataset.apply(
      parsing_ops.parse_example_dataset(
          features, num_parallel_calls=parser_num_threads))
  if label_key:
    if label_key not in features:
      raise ValueError(
          f"The `label_key` provided ({label_key}) must be one of the "
          f"`features` keys: {features.keys()}.")
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))
  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset
