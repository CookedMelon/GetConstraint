@tf_export("summary.record_if", v1=[])
@tf_contextlib.contextmanager
def record_if(condition):
  """Sets summary recording on or off per the provided boolean value.
  The provided value can be a python boolean, a scalar boolean Tensor, or
  or a callable providing such a value; if a callable is passed it will be
  invoked on-demand to determine whether summary writing will occur.  Note that
  when calling record_if() in an eager mode context, if you intend to provide a
  varying condition like `step % 100 == 0`, you must wrap this in a
  callable to avoid immediate eager evaluation of the condition.  In particular,
  using a callable is the only way to have your condition evaluated as part of
  the traced body of an @tf.function that is invoked from within the
  `record_if()` context.
  Args:
    condition: can be True, False, a bool Tensor, or a callable providing such.
  Yields:
    Returns a context manager that sets this value on enter and restores the
    previous value on exit.
  """
  old = _summary_state.is_recording
  try:
    _summary_state.is_recording = condition
    yield
  finally:
    _summary_state.is_recording = old
