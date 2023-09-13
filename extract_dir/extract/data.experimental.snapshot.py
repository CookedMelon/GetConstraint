@tf_export("data.experimental.snapshot")
def snapshot(path, compression="AUTO", reader_func=None, shard_func=None):
  """API to persist the output of the input dataset.
  The snapshot API allows users to transparently persist the output of their
  preprocessing pipeline to disk, and materialize the pre-processed data on a
  different training run.
  This API enables repeated preprocessing steps to be consolidated, and allows
  re-use of already processed data, trading off disk storage and network
  bandwidth for freeing up more valuable CPU resources and accelerator compute
  time.
  https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md
  has detailed design documentation of this feature.
  Users can specify various options to control the behavior of snapshot,
  including how snapshots are read from and written to by passing in
  user-defined functions to the `reader_func` and `shard_func` parameters.
  `shard_func` is a user specified function that maps input elements to snapshot
  shards.
  Users may want to specify this function to control how snapshot files should
  be written to disk. Below is an example of how a potential shard_func could
  be written.
  ```python
  dataset = ...
  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.Dataset.shapshot("/path/to/snapshot/dir",
      shard_func=lambda x, y: x % NUM_SHARDS, ...))
  dataset = dataset.map(lambda x, y: y)
  ```
  `reader_func` is a user specified function that accepts a single argument:
  (1) a Dataset of Datasets, each representing a "split" of elements of the
  original dataset. The cardinality of the input dataset matches the
  number of the shards specified in the `shard_func` (see above). The function
  should return a Dataset of elements of the original dataset.
  Users may want specify this function to control how snapshot files should be
  read from disk, including the amount of shuffling and parallelism.
  Here is an example of a standard reader function a user can define. This
  function enables both dataset shuffling and parallel reading of datasets:
  ```python
  def user_reader_func(datasets):
    # shuffle the datasets splits
    datasets = datasets.shuffle(NUM_CORES)
    # read datasets in parallel and interleave their elements
    return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)
  dataset = dataset.apply(tf.data.Dataset.shapshot("/path/to/snapshot/dir",
      reader_func=user_reader_func))
  ```
  By default, snapshot parallelizes reads by the number of cores available on
  the system, but will not attempt to shuffle the data.
  Args:
    path: Required. A directory to use for storing / loading the snapshot to /
      from.
    compression: Optional. The type of compression to apply to the snapshot
      written to disk. Supported options are `GZIP`, `SNAPPY`, `AUTO` or None.
      Defaults to AUTO, which attempts to pick an appropriate compression
      algorithm for the dataset.
    reader_func: Optional. A function to control how to read data from snapshot
      shards.
    shard_func: Optional. A function to control how to shard data when writing a
      snapshot.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    """Actual dataset transformation."""
    return dataset.snapshot(
        path=path,
        compression=compression,
        reader_func=reader_func,
        shard_func=shard_func)
  return _apply_fn
