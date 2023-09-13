@tf_export("errors.DataLossError")
class DataLossError(OpError):
  """Raised when unrecoverable data loss or corruption is encountered.
  This could be due to:
  * A truncated file.
  * A corrupted file.
  * Specifying the wrong data format.
  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS, *args)
