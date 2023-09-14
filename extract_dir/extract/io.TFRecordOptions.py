@tf_export(
    "io.TFRecordOptions",
    v1=["io.TFRecordOptions", "python_io.TFRecordOptions"])
@deprecation.deprecated_endpoints("python_io.TFRecordOptions")
class TFRecordOptions(object):
  """Options used for manipulating TFRecord files."""
  compression_type_map = {
      TFRecordCompressionType.ZLIB: "ZLIB",
      TFRecordCompressionType.GZIP: "GZIP",
      TFRecordCompressionType.NONE: ""
  }
  def __init__(self,
               compression_type=None,
               flush_mode=None,
               input_buffer_size=None,
               output_buffer_size=None,
               window_bits=None,
               compression_level=None,
               compression_method=None,
               mem_level=None,
               compression_strategy=None):
    # pylint: disable=line-too-long
    """Creates a `TFRecordOptions` instance.
    Options only effect TFRecordWriter when compression_type is not `None`.
    Documentation, details, and defaults can be found in
    [`zlib_compression_options.h`](https://www.tensorflow.org/code/tensorflow/core/lib/io/zlib_compression_options.h)
    and in the [zlib manual](http://www.zlib.net/manual.html).
    Leaving an option as `None` allows C++ to set a reasonable default.
    Args:
      compression_type: `"GZIP"`, `"ZLIB"`, or `""` (no compression).
      flush_mode: flush mode or `None`, Default: Z_NO_FLUSH.
      input_buffer_size: int or `None`.
      output_buffer_size: int or `None`.
      window_bits: int or `None`.
      compression_level: 0 to 9, or `None`.
      compression_method: compression method or `None`.
      mem_level: 1 to 9, or `None`.
      compression_strategy: strategy or `None`. Default: Z_DEFAULT_STRATEGY.
    Returns:
      A `TFRecordOptions` object.
    Raises:
      ValueError: If compression_type is invalid.
    """
    # pylint: enable=line-too-long
    # Check compression_type is valid, but for backwards compatibility don't
    # immediately convert to a string.
    self.get_compression_type_string(compression_type)
    self.compression_type = compression_type
    self.flush_mode = flush_mode
    self.input_buffer_size = input_buffer_size
    self.output_buffer_size = output_buffer_size
    self.window_bits = window_bits
    self.compression_level = compression_level
    self.compression_method = compression_method
    self.mem_level = mem_level
    self.compression_strategy = compression_strategy
  @classmethod
  def get_compression_type_string(cls, options):
    """Convert various option types to a unified string.
    Args:
      options: `TFRecordOption`, `TFRecordCompressionType`, or string.
    Returns:
      Compression type as string (e.g. `'ZLIB'`, `'GZIP'`, or `''`).
    Raises:
      ValueError: If compression_type is invalid.
    """
    if not options:
      return ""
    elif isinstance(options, TFRecordOptions):
      return cls.get_compression_type_string(options.compression_type)
    elif isinstance(options, TFRecordCompressionType):
      return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map:
      return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map.values():
      return options
    else:
      raise ValueError('Not a valid compression_type: "{}"'.format(options))
  def _as_record_writer_options(self):
    """Convert to RecordWriterOptions for use with PyRecordWriter."""
    options = _pywrap_record_io.RecordWriterOptions(
        compat.as_bytes(
            self.get_compression_type_string(self.compression_type)))
    if self.flush_mode is not None:
      options.zlib_options.flush_mode = self.flush_mode
    if self.input_buffer_size is not None:
      options.zlib_options.input_buffer_size = self.input_buffer_size
    if self.output_buffer_size is not None:
      options.zlib_options.output_buffer_size = self.output_buffer_size
    if self.window_bits is not None:
      options.zlib_options.window_bits = self.window_bits
    if self.compression_level is not None:
      options.zlib_options.compression_level = self.compression_level
    if self.compression_method is not None:
      options.zlib_options.compression_method = self.compression_method
    if self.mem_level is not None:
      options.zlib_options.mem_level = self.mem_level
    if self.compression_strategy is not None:
      options.zlib_options.compression_strategy = self.compression_strategy
    return options
@tf_export(v1=["io.tf_record_iterator", "python_io.tf_record_iterator"])
@deprecation.deprecated(
    date=None,
    instructions=("Use eager execution and: \n"
                  "`tf.data.TFRecordDataset(path)`"))
def tf_record_iterator(path, options=None):
  """An iterator that read the records from a TFRecords file.
  Args:
    path: The path to the TFRecords file.
    options: (optional) A TFRecordOptions object.
  Returns:
    An iterator of serialized TFRecords.
  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  compression_type = TFRecordOptions.get_compression_type_string(options)
  return _pywrap_record_io.RecordIterator(path, compression_type)
def tf_record_random_reader(path):
  """Creates a reader that allows random-access reads from a TFRecords file.
  The created reader object has the following method:
    - `read(offset)`, which returns a tuple of `(record, ending_offset)`, where
      `record` is the TFRecord read at the offset, and
      `ending_offset` is the ending offset of the read record.
      The method throws a `tf.errors.DataLossError` if data is corrupted at
      the given offset. The method throws `IndexError` if the offset is out of
      range for the TFRecords file.
  Usage example:
  ```py
  reader = tf_record_random_reader(file_path)
  record_1, offset_1 = reader.read(0)  # 0 is the initial offset.
  # offset_1 is the ending offset of the 1st record and the starting offset of
  # the next.
  record_2, offset_2 = reader.read(offset_1)
  # offset_2 is the ending offset of the 2nd record and the starting offset of
  # the next.
  # We can jump back and read the first record again if so desired.
  reader.read(0)
  ```
  Args:
    path: The path to the TFRecords file.
  Returns:
    An object that supports random-access reading of the serialized TFRecords.
  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  return _pywrap_record_io.RandomRecordReader(path)
