@tf_export("summary.create_noop_writer", v1=[])
def create_noop_writer():
  """Returns a summary writer that does nothing.
  This is useful as a placeholder in code that expects a context manager.
  """
  return _NoopSummaryWriter()
