@tf_export("summary.flush", v1=[])
def flush(writer=None, name=None):
  """Forces summary writer to send any buffered data to storage.
  This operation blocks until that finishes.
  Args:
    writer: The `tf.summary.SummaryWriter` to flush. If None, the current
      default writer will be used instead; if there is no current writer, this
      returns `tf.no_op`.
    name: Ignored legacy argument for a name for the operation.
  Returns:
    The created `tf.Operation`.
  """
  del name  # unused
  if writer is None:
    writer = _summary_state.writer
    if writer is None:
      return control_flow_ops.no_op()
  if isinstance(writer, SummaryWriter):
    return writer.flush()
  raise ValueError("Invalid argument to flush(): %r" % (writer,))
