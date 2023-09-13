@tf_export("errors.UnauthenticatedError")
class UnauthenticatedError(OpError):
  """Raised when the request does not have valid authentication credentials.
  This exception is not currently used.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(node_def, op, message,
                                               UNAUTHENTICATED, *args)
