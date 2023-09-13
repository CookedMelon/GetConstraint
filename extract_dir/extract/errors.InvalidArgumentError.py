@tf_export("errors.InvalidArgumentError")
class InvalidArgumentError(OpError):
  """Raised when an operation receives an invalid argument.
  This error is typically raised when an op receives mismatched arguments.
  Example:
  >>> tf.reshape([1, 2, 3], (2,))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT, *args)
