@tf_export("errors.UnimplementedError")
class UnimplementedError(OpError):
  """Raised when an operation has not been implemented.
  Some operations may raise this error when passed otherwise-valid
  arguments that it does not currently support. For example, running
  the `tf.nn.max_pool2d` operation
  would raise this error if pooling was requested on the batch dimension,
  because this is not yet supported.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(node_def, op, message,
                                             UNIMPLEMENTED, *args)
