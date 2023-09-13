"/home/cc/Workspace/tfconstraint/python/lib/io/tf_record.py"
@tf_export(
    "io.TFRecordWriter", v1=["io.TFRecordWriter", "python_io.TFRecordWriter"])
@deprecation.deprecated_endpoints("python_io.TFRecordWriter")
class TFRecordWriter(_pywrap_record_io.RecordWriter):
  """A class to write records to a TFRecords file.
  [TFRecords tutorial](https://www.tensorflow.org/tutorials/load_data/tfrecord)
  TFRecords is a binary format which is optimized for high throughput data
  retrieval, generally in conjunction with `tf.data`. `TFRecordWriter` is used
  to write serialized examples to a file for later consumption. The key steps
  are:
   Ahead of time:
   - [Convert data into a serialized format](
   https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample)
   - [Write the serialized data to one or more files](
   https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecord_files_in_python)
   During training or evaluation:
   - [Read serialized examples into memory](
   https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file)
   - [Parse (deserialize) examples](
   https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file)
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
  This class implements `__enter__` and `__exit__`, and can be used
  in `with` blocks like a normal file. (See the usage example above.)
  """
  # TODO(josh11b): Support appending?
  def __init__(self, path, options=None):
    """Opens file `path` and creates a `TFRecordWriter` writing to it.
    Args:
      path: The path to the TFRecords file.
      options: (optional) String specifying compression type,
          `TFRecordCompressionType`, or `TFRecordOptions` object.
    Raises:
      IOError: If `path` cannot be opened for writing.
      ValueError: If valid compression_type can't be determined from `options`.
    """
    if not isinstance(options, TFRecordOptions):
      options = TFRecordOptions(compression_type=options)
    # pylint: disable=protected-access
    super(TFRecordWriter, self).__init__(
        compat.as_bytes(path), options._as_record_writer_options())
    # pylint: enable=protected-access
  # TODO(slebedev): The following wrapper methods are there to compensate
  # for lack of signatures in pybind11-generated classes. Switch to
  # __text_signature__ when TensorFlow drops Python 2.X support.
  # See https://github.com/pybind/pybind11/issues/945
  # pylint: disable=useless-super-delegation
  def write(self, record):
    """Write a string record to the file.
    Args:
      record: str
    """
    super(TFRecordWriter, self).write(record)
  def flush(self):
    """Flush the file."""
    super(TFRecordWriter, self).flush()
  def close(self):
    """Close the file."""
    super(TFRecordWriter, self).close()
  # pylint: enable=useless-super-delegation
