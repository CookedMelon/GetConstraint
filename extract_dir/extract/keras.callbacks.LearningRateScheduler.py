@keras_export("keras.callbacks.LearningRateScheduler")
class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.
    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
          and current learning rate (float) as inputs and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
    Example:
    >>> # This function keeps the initial learning rate for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, lr):
    ...   if epoch < 10:
    ...     return lr
    ...   else:
    ...     return lr * tf.math.exp(-0.1)
    >>>
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> round(model.optimizer.lr.numpy(), 5)
    0.01
    >>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.optimizer.lr.numpy(), 5)
    0.00607
    """
    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)
