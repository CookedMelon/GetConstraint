@tf_export("data.TextLineDataset", v1=[])
class TextLineDatasetV2(dataset_ops.DatasetSource):
  r"""Creates a `Dataset` comprising lines from one or more text files.
  The `tf.data.TextLineDataset` loads text from text files and creates a dataset
  where each line of the files becomes an element of the dataset.
  For example, suppose we have 2 files "text_lines0.txt" and "text_lines1.txt"
  with the following lines:
  >>> with open('/tmp/text_lines0.txt', 'w') as f:
  ...   f.write('the cow\n')
  ...   f.write('jumped over\n')
  ...   f.write('the moon\n')
  >>> with open('/tmp/text_lines1.txt', 'w') as f:
  ...   f.write('jack and jill\n')
  ...   f.write('went up\n')
  ...   f.write('the hill\n')
  We can construct a TextLineDataset from them as follows:
  >>> dataset = tf.data.TextLineDataset(['/tmp/text_lines0.txt',
  ...                                    '/tmp/text_lines1.txt'])
  The elements of the dataset are expected to be:
  >>> for element in dataset.as_numpy_iterator():
  ...   print(element)
  b'the cow'
  b'jumped over'
  b'the moon'
  b'jack and jill'
  b'went up'
  b'the hill'
  """
  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    r"""Creates a `TextLineDataset`.
    The elements of the dataset will be the lines of the input files, using
    the newline character '\n' to denote line splits. The newline characters
    will be stripped off of each element.
    Args:
      filenames: A `tf.data.Dataset` whose elements are `tf.string` scalars, a
        `tf.string` tensor, or a value that can be converted to a `tf.string`
        tensor (such as a list of Python strings).
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
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
    self._compression_type = compression_type
    self._buffer_size = buffer_size
    def creator_fn(filename):
      return _TextLineDataset(
          filename, compression_type, buffer_size, name=name)
    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(TextLineDatasetV2, self).__init__(variant_tensor)
  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)
