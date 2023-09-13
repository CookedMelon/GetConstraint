@tf_export("errors.ResourceExhaustedError")
class ResourceExhaustedError(OpError):
  """Raised when some resource has been exhausted while running operation.
  For example, this error might be raised if a per-user quota is
  exhausted, or perhaps the entire file system is out of space. If running into
  `ResourceExhaustedError` due to out of memory (OOM), try to use smaller batch
  size or reduce dimension size of model weights.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(node_def, op, message,
                                                 RESOURCE_EXHAUSTED, *args)
