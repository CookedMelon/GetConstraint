@tf_export("summary.experimental.get_step", v1=[])
def get_step():
  """Returns the default summary step for the current thread.
  Returns:
    The step set by `tf.summary.experimental.set_step()` if one has been set,
    otherwise None.
  """
  return _summary_state.step
