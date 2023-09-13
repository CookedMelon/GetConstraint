@tf_export("debugging.disable_check_numerics")
def disable_check_numerics():
  """Disable the eager/graph unified numerics checking mechanism.
  This method can be used after a call to `tf.debugging.enable_check_numerics()`
  to disable the numerics-checking mechanism that catches infinity and NaN
  values output by ops executed eagerly or in tf.function-compiled graphs.
  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.
  This method takes effect only on the thread in which it is called.
  """
  if not hasattr(_state, "check_numerics_callback"):
    return
  try:
    op_callbacks.remove_op_callback(_state.check_numerics_callback.callback)
    delattr(_state, "check_numerics_callback")
    logging.info(
        "Disabled check-numerics callback in thread %s",
        threading.current_thread().name)
  except KeyError:
    # Tolerate disabling the check numerics callback without
    # enable_check_numerics() being called first.
    pass
