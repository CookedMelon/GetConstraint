@tf_export("summary.experimental.set_step", v1=[])
def set_step(step):
  """Sets the default summary step for the current thread.
  For convenience, this function sets a default value for the `step` parameter
  used in summary-writing functions elsewhere in the API so that it need not
  be explicitly passed in every such invocation. The value can be a constant
  or a variable, and can be retrieved via `tf.summary.experimental.get_step()`.
  Note: when using this with @tf.functions, the step value will be captured at
  the time the function is traced, so changes to the step outside the function
  will not be reflected inside the function unless using a `tf.Variable` step.
  Args:
    step: An `int64`-castable default step value, or None to unset.
  """
  _summary_state.step = step
