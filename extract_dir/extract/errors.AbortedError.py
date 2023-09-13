@tf_export("errors.AbortedError")
class AbortedError(OpError):
  """Raised when an operation was aborted, typically due to a concurrent action.
  For example, running a
  `tf.queue.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.queue.QueueBase.close` operation
  previously ran.
  """
  def __init__(self, node_def, op, message, *args):
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(node_def, op, message, ABORTED, *args)
