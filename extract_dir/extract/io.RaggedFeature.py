@tf_export("io.RaggedFeature")
class RaggedFeature(
    collections.namedtuple(
        "RaggedFeature",
        ["dtype", "value_key", "partitions", "row_splits_dtype", "validate"])):
  """Configuration for passing a RaggedTensor input feature.
  `value_key` specifies the feature key for a variable-length list of values;
  and `partitions` specifies zero or more feature keys for partitioning those
  values into higher dimensions.  Each element of `partitions` must be one of
  the following:
    * `tf.io.RaggedFeature.RowSplits(key: string)`
    * `tf.io.RaggedFeature.RowLengths(key: string)`
    * `tf.io.RaggedFeature.RowStarts(key: string)`
    * `tf.io.RaggedFeature.RowLimits(key: string)`
    * `tf.io.RaggedFeature.ValueRowIds(key: string)`
    * `tf.io.RaggedFeature.UniformRowLength(length: int)`.
  Where `key` is a feature key whose values are used to partition the values.
  Partitions are listed from outermost to innermost.
  * If `len(partitions) == 0` (the default), then:
    * A feature from a single `tf.Example` is parsed into a 1D `tf.Tensor`.
    * A feature from a batch of `tf.Example`s is parsed into a 2D
      `tf.RaggedTensor`, where the outer dimension is the batch dimension, and
      the inner (ragged) dimension is the feature length in each example.
  * If `len(partitions) == 1`, then:
    * A feature from a single `tf.Example` is parsed into a 2D
      `tf.RaggedTensor`, where the values taken from the `value_key` are
      separated into rows using the partition key.
    * A feature from a batch of `tf.Example`s is parsed into a 3D
      `tf.RaggedTensor`, where the outer dimension is the batch dimension,
      the two inner dimensions are formed by separating the `value_key` values
      from each example into rows using that example's partition key.
  * If `len(partitions) > 1`, then:
    * A feature from a single `tf.Example` is parsed into a `tf.RaggedTensor`
      whose rank is `len(partitions)+1`, and whose ragged_rank is
      `len(partitions)`.
    * A feature from a batch of `tf.Example`s is parsed into a `tf.RaggedTensor`
      whose rank is `len(partitions)+2` and whose ragged_rank is
      `len(partitions)+1`, where the outer dimension is the batch dimension.
  There is one exception: if the final (i.e., innermost) element(s) of
  `partitions` are `UniformRowLength`s, then the values are simply reshaped (as
  a higher-dimensional `tf.Tensor`), rather than being wrapped in a
  `tf.RaggedTensor`.
  #### Examples
  >>> import google.protobuf.text_format as pbtext
  >>> example_batch = [
  ...   pbtext.Merge(r'''
  ...     features {
  ...       feature {key: "v" value {int64_list {value: [3, 1, 4, 1, 5, 9]}}}
  ...       feature {key: "s1" value {int64_list {value: [0, 2, 3, 3, 6]}}}
  ...       feature {key: "s2" value {int64_list {value: [0, 2, 3, 4]}}}
  ...     }''', tf.train.Example()).SerializeToString(),
  ...   pbtext.Merge(r'''
  ...     features {
  ...       feature {key: "v" value {int64_list {value: [2, 7, 1, 8, 2, 8, 1]}}}
  ...       feature {key: "s1" value {int64_list {value: [0, 3, 4, 5, 7]}}}
  ...       feature {key: "s2" value {int64_list {value: [0, 1, 1, 4]}}}
  ...     }''', tf.train.Example()).SerializeToString()]
  >>> features = {
  ...     # Zero partitions: returns 1D tf.Tensor for each Example.
  ...     'f1': tf.io.RaggedFeature(value_key="v", dtype=tf.int64),
  ...     # One partition: returns 2D tf.RaggedTensor for each Example.
  ...     'f2': tf.io.RaggedFeature(value_key="v", dtype=tf.int64, partitions=[
  ...         tf.io.RaggedFeature.RowSplits("s1")]),
  ...     # Two partitions: returns 3D tf.RaggedTensor for each Example.
  ...     'f3': tf.io.RaggedFeature(value_key="v", dtype=tf.int64, partitions=[
  ...         tf.io.RaggedFeature.RowSplits("s2"),
  ...         tf.io.RaggedFeature.RowSplits("s1")])
  ... }
  >>> feature_dict = tf.io.parse_single_example(example_batch[0], features)
  >>> for (name, val) in sorted(feature_dict.items()):
  ...   print('%s: %s' % (name, val))
  f1: tf.Tensor([3 1 4 1 5 9], shape=(6,), dtype=int64)
  f2: <tf.RaggedTensor [[3, 1], [4], [], [1, 5, 9]]>
  f3: <tf.RaggedTensor [[[3, 1], [4]], [[]], [[1, 5, 9]]]>
  >>> feature_dict = tf.io.parse_example(example_batch, features)
  >>> for (name, val) in sorted(feature_dict.items()):
  ...   print('%s: %s' % (name, val))
  f1: <tf.RaggedTensor [[3, 1, 4, 1, 5, 9],
                        [2, 7, 1, 8, 2, 8, 1]]>
  f2: <tf.RaggedTensor [[[3, 1], [4], [], [1, 5, 9]],
                        [[2, 7, 1], [8], [2], [8, 1]]]>
  f3: <tf.RaggedTensor [[[[3, 1], [4]], [[]], [[1, 5, 9]]],
                        [[[2, 7, 1]], [], [[8], [2], [8, 1]]]]>
  Fields:
    dtype: Data type of the `RaggedTensor`.  Must be one of:
      `tf.dtypes.int64`, `tf.dtypes.float32`, `tf.dtypes.string`.
    value_key: (Optional.) Key for a `Feature` in the input `Example`, whose
      parsed `Tensor` will be the resulting `RaggedTensor.flat_values`.  If
      not specified, then it defaults to the key for this `RaggedFeature`.
    partitions: (Optional.) A list of objects specifying the row-partitioning
      tensors (from outermost to innermost).  Each entry in this list must be
      one of:
        * `tf.io.RaggedFeature.RowSplits(key: string)`
        * `tf.io.RaggedFeature.RowLengths(key: string)`
        * `tf.io.RaggedFeature.RowStarts(key: string)`
        * `tf.io.RaggedFeature.RowLimits(key: string)`
        * `tf.io.RaggedFeature.ValueRowIds(key: string)`
        * `tf.io.RaggedFeature.UniformRowLength(length: int)`.
      Where `key` is a key for a `Feature` in the input `Example`, whose parsed
      `Tensor` will be the resulting row-partitioning tensor.
    row_splits_dtype: (Optional.) Data type for the row-partitioning tensor(s).
      One of `int32` or `int64`.  Defaults to `int32`.
    validate: (Optional.) Boolean indicating whether or not to validate that
      the input values form a valid RaggedTensor.  Defaults to `False`.
  """
  # pylint: disable=invalid-name
  RowSplits = collections.namedtuple("RowSplits", ["key"])
  RowLengths = collections.namedtuple("RowLengths", ["key"])
  RowStarts = collections.namedtuple("RowStarts", ["key"])
  RowLimits = collections.namedtuple("RowLimits", ["key"])
  ValueRowIds = collections.namedtuple("ValueRowIds", ["key"])
  UniformRowLength = collections.namedtuple("UniformRowLength", ["length"])
  # pylint: enable=invalid-name
  _PARTITION_TYPES = (RowSplits, RowLengths, RowStarts, RowLimits, ValueRowIds,
                      UniformRowLength)
  def __new__(cls,
              dtype,
              value_key=None,
              partitions=(),
              row_splits_dtype=dtypes.int32,
              validate=False):
    if value_key is not None:
      if not isinstance(value_key, str):
        raise ValueError(
            f"Argument `value_key` must be a string; got {value_key}")
      if not value_key:
        raise ValueError("Argument `value_key` must not be empty")
    dtype = dtypes.as_dtype(dtype)
    if dtype not in (dtypes.int64, dtypes.float32, dtypes.string):
      raise ValueError("Argument `dtype` must be int64, float32, or bytes; got "
                       f"{dtype!r}")
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if row_splits_dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("Argument `row_splits_dtype` must be int32 or int64; got"
                       f"{row_splits_dtype!r}")
    if not isinstance(partitions, (list, tuple)):
      raise TypeError("Argument `partitions` must be a list or tuple. Received"
                      f"partitions={partitions} of type "
                      f"{type(partitions).__name__}.")
    for partition in partitions:
      if not isinstance(partition, cls._PARTITION_TYPES):
        raise TypeError("Argument `partitions` must be a list of partition "
                        f"objects {cls._PARTITION_TYPES}; got: {partition!r}")
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must be a bool; got {validate!r}")
    return super(RaggedFeature, cls).__new__(cls, dtype, value_key, partitions,
                                             row_splits_dtype, validate)
