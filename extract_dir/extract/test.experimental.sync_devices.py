@tf_export("test.experimental.sync_devices")
def sync_devices():
  """Synchronizes all devices.
  By default, GPUs run asynchronously. This means that when you run an op on the
  GPU, like `tf.linalg.matmul`, the op may still be running on the GPU when the
  function returns. Non-GPU devices can also be made to run asynchronously by
  calling `tf.config.experimental.set_synchronous_execution(False)`. Calling
  `sync_devices()` blocks until pending ops have finished executing. This is
  primarily useful for measuring performance during a benchmark.
  For example, here is how you can measure how long `tf.linalg.matmul` runs:
  >>> import time
  >>> x = tf.random.normal((4096, 4096))
  >>> tf.linalg.matmul(x, x)  # Warmup.
  >>> tf.test.experimental.sync_devices()  # Block until warmup has completed.
  >>>
  >>> start = time.time()
  >>> y = tf.linalg.matmul(x, x)
  >>> tf.test.experimental.sync_devices()  # Block until matmul has completed.
  >>> end = time.time()
  >>> print(f'Time taken: {end - start}')
  If the call to `sync_devices()` was omitted, the time printed could be too
  small. This is because the op could still be running asynchronously when
  the line `end = time.time()` is executed.
  Raises:
    RuntimeError: If run outside Eager mode. This must be called in Eager mode,
      outside any `tf.function`s.
  """
  if not context.executing_eagerly():
    raise RuntimeError(
        "sync_devices() must only be called in Eager mode, outside tf.functions"
    )
  # There are two sources of asynchrony in TensorFlow:
  #
  # 1. On GPUs, kernels are run on a CUDA stream, which is inherently
  #    asynchronous.
  # 2. Calling `tf.config.experimental.set_synchronous_execution(False)` makes
  #    all ops asynchronous, in which case TensorFlow maintains internal queues
  #    of pending ops.
  #
  # Calling SyncDevice addresses source (1). Calling async_await addresses
  # source (2). It is important that SyncDevice() is called before async_wait(),
  # otherwise the SyncDevice op itself may still be pending on an internal
  # TensorFlow queue when the sync_devices() Python function returns.
  devices = config.list_logical_devices()
  for dev in devices:
    with ops.device(dev.name):
      gen_sync_ops.SyncDevice()
  context.async_wait()
