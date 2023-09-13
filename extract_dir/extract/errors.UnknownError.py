@tf_export("errors.UnknownError")
class UnknownError(OpError):
  """Unknown error.
  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also, errors raised by APIs that
  do not return enough error information may be converted to this
  error.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(node_def, op, message, UNKNOWN, *args)
