@tf_export("data.experimental.save", v1=[])
@deprecation.deprecated(None, "Use `tf.data.Dataset.save(...)` instead.")
def save(dataset,
         path,
         compression=None,
         shard_func=None,
         checkpoint_args=None):
  """Saves the content of the given dataset.
  Example usage:
  >>> import tempfile
  >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
  >>> # Save a dataset
  >>> dataset = tf.data.Dataset.range(2)
  >>> tf.data.experimental.save(dataset, path)
  >>> new_dataset = tf.data.experimental.load(path)
  >>> for elem in new_dataset:
  ...   print(elem)
  tf.Tensor(0, shape=(), dtype=int64)
  tf.Tensor(1, shape=(), dtype=int64)
  The saved dataset is saved in multiple file "shards". By default, the dataset
  output is divided to shards in a round-robin fashion but custom sharding can
  be specified via the `shard_func` function. For example, you can save the
  dataset to using a single shard as follows:
  ```python
  dataset = make_dataset()
  def custom_shard_func(element):
    return np.int64(0)
  dataset = tf.data.experimental.save(
      path="/path/to/data", ..., shard_func=custom_shard_func)
  ```
  To enable checkpointing, pass in `checkpoint_args` to the `save` method
  as follows:
  ```python
  dataset = tf.data.Dataset.range(100)
  save_dir = "..."
  checkpoint_prefix = "..."
  step_counter = tf.Variable(0, trainable=False)
  checkpoint_args = {
    "checkpoint_interval": 50,
    "step_counter": step_counter,
    "directory": checkpoint_prefix,
    "max_to_keep": 20,
  }
  dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)
  ```
  NOTE: The directory layout and file format used for saving the dataset is
  considered an implementation detail and may change. For this reason, datasets
  saved through `tf.data.experimental.save` should only be consumed through
  `tf.data.experimental.load`, which is guaranteed to be backwards compatible.
  Args:
    dataset: The dataset to save.
    path: Required. A directory to use for saving the dataset.
    compression: Optional. The algorithm to use to compress data when writing
      it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
    shard_func: Optional. A function to control the mapping of dataset elements
      to file shards. The function is expected to map elements of the input
      dataset to int64 shard IDs. If present, the function will be traced and
      executed as graph computation.
    checkpoint_args: Optional args for checkpointing which will be passed into
      the `tf.train.CheckpointManager`. If `checkpoint_args` are not specified,
      then checkpointing will not be performed. The `save()` implementation
      creates a `tf.train.Checkpoint` object internally, so users should not
      set the `checkpoint` argument in `checkpoint_args`.
  Returns:
    An operation which when executed performs the save. When writing
    checkpoints, returns None. The return value is useful in unit tests.
  Raises:
    ValueError if `checkpoint` is passed into `checkpoint_args`.
  """
  return dataset.save(path, compression, shard_func, checkpoint_args)
