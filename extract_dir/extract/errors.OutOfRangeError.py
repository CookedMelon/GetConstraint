@tf_export("errors.OutOfRangeError")
class OutOfRangeError(OpError):
  """Raised when an operation iterates past the valid range.
  Unlike `InvalidArgumentError`, this error indicates a problem may be fixed if
  the system state changes. For example, if a list grows and the operation is
  now within the valid range. `OutOfRangeError` overlaps with
  `FailedPreconditionError` and should be preferred as the more specific error
  when iterating or accessing a range.
  For example, iterating a TF dataset past the last item in the dataset will
  raise this error.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(node_def, op, message, OUT_OF_RANGE,
                                          *args)
