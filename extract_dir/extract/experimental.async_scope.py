@tf_export("experimental.async_scope")
@tf_contextlib.contextmanager
def async_scope():
  """Context manager for grouping async operations.
  Ops/function calls inside the scope can return before finishing the actual
  execution. When exiting the async scope, a synchronization barrier will be
  automatically added to ensure the completion of all async op and function
  execution, potentially raising exceptions if async execution results in
  an error state.
  Users may write the following code to asynchronously invoke `train_step_fn`
  and log the `loss` metric for every `num_steps` steps in a training loop.
  `train_step_fn` internally consumes data using `iterator.get_next()`, and may
  throw OutOfRangeError when running out of data. In the case:
  ```
  try:
    with tf.experimental.async_scope():
      for _ in range(num_steps):
        # Step function updates the metric `loss` internally
        train_step_fn()
  except tf.errors.OutOfRangeError:
    tf.experimental.async_clear_error()
  logging.info('loss = %s', loss.numpy())
  ```
  Yields:
    Context manager for grouping async operations.
  """
  # TODO(haoyuzhang): replace env var once we have a config method to turn on
  # and off async streaming RPC
  remote_async_env_var = "TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"
  old_policy = os.environ.get(remote_async_env_var)
  try:
    os.environ[remote_async_env_var] = str(True)
    yield
    # Note: sync local and remote executors iff the async block does not raise
    # an exception. Triggering sync after an exception may lead to derived
    # runtime errors and unexpected exception types.
    context().sync_executors()
  finally:
    if old_policy is None:
      del os.environ[remote_async_env_var]
    else:
      os.environ[remote_async_env_var] = old_policy
