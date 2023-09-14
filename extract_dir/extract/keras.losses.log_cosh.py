@keras_export(
    "keras.losses.log_cosh",
    "keras.losses.logcosh",
    "keras.metrics.log_cosh",
    "keras.metrics.logcosh",
)
@tf.__internal__.dispatch.add_dispatch_support
def log_cosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    Standalone usage:
    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> x = y_pred - y_true
    >>> assert np.allclose(
    ...     loss.numpy(),
    ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - tf.math.log(2.),
    ...             axis=-1),
    ...     atol=1e-5)
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    def _logcosh(x):
        return (
            x + tf.math.softplus(-2.0 * x) - tf.cast(tf.math.log(2.0), x.dtype)
        )
    return backend.mean(_logcosh(y_pred - y_true), axis=-1)
