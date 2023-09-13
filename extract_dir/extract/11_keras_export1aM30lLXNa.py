"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.kl_divergence",
    "keras.metrics.kullback_leibler_divergence",
    "keras.metrics.kld",
    "keras.metrics.KLD",
    "keras.losses.kl_divergence",
    "keras.losses.kullback_leibler_divergence",
    "keras.losses.kld",
    "keras.losses.KLD",
)
@tf.__internal__.dispatch.add_dispatch_support
def kl_divergence(y_true, y_pred):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.
    `loss = y_true * log(y_true / y_pred)`
    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
    >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))
    Args:
      y_true: Tensor of true targets.
      y_pred: Tensor of predicted targets.
    Returns:
      A `Tensor` with loss.
    Raises:
      TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = backend.clip(y_true, backend.epsilon(), 1)
    y_pred = backend.clip(y_pred, backend.epsilon(), 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)
