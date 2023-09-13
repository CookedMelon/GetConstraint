@tf_export("errors.NotFoundError")
class NotFoundError(OpError):
  """Raised when a requested entity (e.g., a file or directory) was not found.
  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `NotFoundError` if it receives the name of a file that
  does not exist.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND, *args)
