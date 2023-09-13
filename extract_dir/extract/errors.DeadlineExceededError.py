@tf_export("errors.DeadlineExceededError")
class DeadlineExceededError(OpError):
  """Raised when a deadline expires before an operation could complete.
  This exception is not currently used.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED, *args)
