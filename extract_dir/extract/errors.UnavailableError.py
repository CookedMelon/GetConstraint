@tf_export("errors.UnavailableError")
class UnavailableError(OpError):
  """Raised when the runtime is currently unavailable.
  This exception is not currently used.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(node_def, op, message, UNAVAILABLE,
                                           *args)
