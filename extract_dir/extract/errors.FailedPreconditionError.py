@tf_export("errors.FailedPreconditionError")
class FailedPreconditionError(OpError):
  """Raised when some prerequisites are not met when running an operation.
  This typically indicates that system is not in state to execute the operation
  and requires preconditions to be met before successfully executing current
  operation.
  For example, this exception is commonly raised when running an operation
  that reads a `tf.Variable` before it has been initialized.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(node_def, op, message,
                                                  FAILED_PRECONDITION, *args)
