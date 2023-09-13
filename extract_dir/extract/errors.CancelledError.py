@tf_export("errors.CancelledError")
class CancelledError(OpError):
  """Raised when an operation is cancelled.
  For example, a long-running operation e.g.`tf.queue.QueueBase.enqueue`, or a
  `tf.function` call may be cancelled by either running another operation e.g.
  `tf.queue.QueueBase.close` or a remote worker failure.
  This long-running operation will fail by raising `CancelledError`.
  Example:
  >>> q = tf.queue.FIFOQueue(10, tf.float32, ((),))
  >>> q.enqueue((10.0,))
  >>> q.close()
  >>> q.enqueue((10.0,))
  Traceback (most recent call last):
    ...
  CancelledError: ...
  """
  def __init__(self, node_def, op, message, *args):
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED,
                                         *args)
