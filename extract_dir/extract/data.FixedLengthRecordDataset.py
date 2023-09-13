@tf_export("data.FixedLengthRecordDataset", v1=[])
class FixedLengthRecordDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` of fixed-length records from one or more binary files.
  The `tf.data.FixedLengthRecordDataset` reads fixed length records from binary
  files and creates a dataset where each record becomes an element of the
  dataset. The binary files can have a fixed length header and a fixed length
  footer, which will both be skipped.
  For example, suppose we have 2 files "fixed_length0.bin" and
  "fixed_length1.bin" with the following content:
  >>> with open('/tmp/fixed_length0.bin', 'wb') as f:
  ...   f.write(b'HEADER012345FOOTER')
  >>> with open('/tmp/fixed_length1.bin', 'wb') as f:
  ...   f.write(b'HEADER6789abFOOTER')
  We can construct a `FixedLengthRecordDataset` from them as follows:
  >>> dataset1 = tf.data.FixedLengthRecordDataset(
  ...     filenames=['/tmp/fixed_length0.bin', '/tmp/fixed_length1.bin'],
  ...     record_bytes=2, header_bytes=6, footer_bytes=6)
  The elements of the dataset are:
  >>> for element in dataset1.as_numpy_iterator():
  ...   print(element)
  b'01'
  b'23'
  b'45'
  b'67'
  b'89'
  b'ab'
  """
  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               num_parallel_reads=None,
               name=None):
    """Creates a `FixedLengthRecordDataset`.
    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
      name: (Optional.) A name for the tf.data operation.
    """
    filenames = _create_or_validate_filenames_dataset(filenames, name=name)
    self._filenames = filenames
    self._record_bytes = record_bytes
    self._header_bytes = header_bytes
    self._footer_bytes = footer_bytes
    self._buffer_size = buffer_size
    self._compression_type = compression_type
    def creator_fn(filename):
      return _FixedLengthRecordDataset(
          filename,
          record_bytes,
          header_bytes,
          footer_bytes,
          buffer_size,
          compression_type,
          name=name)
    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(FixedLengthRecordDatasetV2, self).__init__(variant_tensor)
  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)
