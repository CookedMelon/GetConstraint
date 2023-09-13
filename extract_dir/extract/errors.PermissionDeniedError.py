@tf_export("errors.PermissionDeniedError")
class PermissionDeniedError(OpError):
  """Raised when the caller does not have permission to run an operation.
  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `PermissionDeniedError` if it receives the name of a
  file for which the user does not have the read file permission.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(node_def, op, message,
                                                PERMISSION_DENIED, *args)
