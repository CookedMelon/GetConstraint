@tf_export("data.experimental.map_and_batch")
def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None):
  """Fused implementation of `map` and `batch`.
  Maps `map_func` across `batch_size` consecutive elements of this dataset
  and then combines them into a batch. Functionally, it is equivalent to `map`
  followed by `batch`. This API is temporary and deprecated since input pipeline
  optimization now fuses consecutive `map` and `batch` operations automatically.
  Args:
    map_func: A function mapping a nested structure of tensors to another
      nested structure of tensors.
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
      representing the number of batches to create in parallel. On one hand,
      higher values can help mitigate the effect of stragglers. On the other
      hand, higher values can increase contention if CPU is scarce.
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in case its size is smaller than
      desired; the default behavior is not to drop the smaller batch.
    num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of elements to process in parallel. If not
      specified, `batch_size * num_parallel_batches` elements will be processed
      in parallel. If the value `tf.data.AUTOTUNE` is used, then
      the number of parallel calls is set dynamically based on available CPU.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  Raises:
    ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
      specified.
  """
  if num_parallel_batches is None and num_parallel_calls is None:
    num_parallel_calls = batch_size
  elif num_parallel_batches is not None and num_parallel_calls is None:
    num_parallel_calls = batch_size * num_parallel_batches
  elif num_parallel_batches is not None and num_parallel_calls is not None:
    raise ValueError(
        "`map_and_batch` allows only one of `num_parallel_batches` and "
        "`num_parallel_calls` to be set, but "
        f"`num_parallel_batches` was set to {num_parallel_batches} "
        f"and `num_parallel_calls` as set to {num_parallel_calls}.")
  def _apply_fn(dataset):
    return _MapAndBatchDataset(dataset, map_func, batch_size,
                               num_parallel_calls, drop_remainder)
  return _apply_fn
