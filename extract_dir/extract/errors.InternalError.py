@tf_export("errors.InternalError")
class InternalError(OpError):
  """Raised when the system experiences an internal error.
  This exception is raised when some invariant expected by the runtime
  has been broken. Catching this exception is not recommended.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(node_def, op, message, INTERNAL, *args)
