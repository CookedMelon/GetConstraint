@tf_export("data.TFRecordDataset", v1=[])
class TFRecordDatasetV2(dataset_ops.DatasetV2):
  """A `Dataset` comprising records from one or more TFRecord files.
  This dataset loads TFRecords from the files as bytes, exactly as they were
  written.`TFRecordDataset` does not do any parsing or decoding on its own.
  Parsing and decoding can be done by applying `Dataset.map` transformations
  after the `TFRecordDataset`.
  A minimal example is given below:
  >>> import tempfile
  >>> example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
  >>> np.random.seed(0)
  >>> # Write the records to a file.
  ... with tf.io.TFRecordWriter(example_path) as file_writer:
  ...   for _ in range(4):
  ...     x, y = np.random.random(), np.random.random()
  ...
  ...     record_bytes = tf.train.Example(features=tf.train.Features(feature={
  ...         "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
  ...         "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
  ...     })).SerializeToString()
  ...     file_writer.write(record_bytes)
  >>> # Read the data back out.
  >>> def decode_fn(record_bytes):
  ...   return tf.io.parse_single_example(
  ...       # Data
  ...       record_bytes,
  ...
  ...       # Schema
  ...       {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
  ...        "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
  ...   )
  >>> for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
  ...   print("x = {x:.4f},  y = {y:.4f}".format(**batch))
  x = 0.5488,  y = 0.7152
  x = 0.6028,  y = 0.5449
  x = 0.4237,  y = 0.6459
  x = 0.4376,  y = 0.8918
  """
  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    """Creates a `TFRecordDataset` to read one or more TFRecord files.
    Each element of the dataset will contain a single TFRecord.
    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. If your input pipeline is I/O bottlenecked,
        consider setting this parameter to a value 1-100 MBs. If `None`, a
        sensible default for both local and remote file systems is used.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
      name: (Optional.) A name for the tf.data operation.
    Raises:
      TypeError: If any argument does not have the expected type.
      ValueError: If any argument does not have the expected shape.
    """
    filenames = _create_or_validate_filenames_dataset(filenames, name=name)
    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size
    self._num_parallel_reads = num_parallel_reads
    def creator_fn(filename):
      return _TFRecordDataset(
          filename, compression_type, buffer_size, name=name)
    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(TFRecordDatasetV2, self).__init__(variant_tensor)
  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access
  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)
