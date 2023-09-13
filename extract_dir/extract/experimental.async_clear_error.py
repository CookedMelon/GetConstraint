@tf_export("experimental.async_clear_error")
def async_clear_error():
  """Clear pending operations and error statuses in async execution.
  In async execution mode, an error in op/function execution can lead to errors
  in subsequent ops/functions that are scheduled but not yet executed. Calling
  this method clears all pending operations and reset the async execution state.
  Example:
  ```
  while True:
    try:
      # Step function updates the metric `loss` internally
      train_step_fn()
    except tf.errors.OutOfRangeError:
      tf.experimental.async_clear_error()
      break
  logging.info('loss = %s', loss.numpy())
  ```
  """
  context().clear_executor_errors()
