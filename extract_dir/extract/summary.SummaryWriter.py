@tf_export("summary.SummaryWriter", v1=[])
class SummaryWriter(metaclass=abc.ABCMeta):
  """Interface representing a stateful summary writer object."""
  def set_as_default(self, step=None):
    """Enables this summary writer for the current thread.
    For convenience, if `step` is not None, this function also sets a default
    value for the `step` parameter used in summary-writing functions elsewhere
    in the API so that it need not be explicitly passed in every such
    invocation. The value can be a constant or a variable.
    Note: when setting `step` in a @tf.function, the step value will be
    captured at the time the function is traced, so changes to the step outside
    the function will not be reflected inside the function unless using
    a `tf.Variable` step.
    Args:
      step: An `int64`-castable default step value, or `None`. When not `None`,
        the current step is modified to the given value. When `None`, the
        current step is not modified.
    """
    self.as_default(step).__enter__()
  def as_default(self, step=None):
    """Returns a context manager that enables summary writing.
    For convenience, if `step` is not None, this function also sets a default
    value for the `step` parameter used in summary-writing functions elsewhere
    in the API so that it need not be explicitly passed in every such
    invocation. The value can be a constant or a variable.
    Note: when setting `step` in a @tf.function, the step value will be
    captured at the time the function is traced, so changes to the step outside
    the function will not be reflected inside the function unless using
    a `tf.Variable` step.
    For example, `step` can be used as:
    ```python
    with writer_a.as_default(step=10):
      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10
      with writer_b.as_default(step=20):
        tf.summary.scalar(tag, value) # Logged to writer_b with step 20
      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10
    ```
    Args:
      step: An `int64`-castable default step value, or `None`. When not `None`,
        the current step is captured, replaced by a given one, and the original
        one is restored when the context manager exits. When `None`, the current
        step is not modified (and not restored when the context manager exits).
    Returns:
      The context manager.
    """
    return _SummaryContextManager(self, step)
  def init(self):
    """Initializes the summary writer."""
    raise NotImplementedError()
  def flush(self):
    """Flushes any buffered data."""
    raise NotImplementedError()
  def close(self):
    """Flushes and closes the summary writer."""
    raise NotImplementedError()
