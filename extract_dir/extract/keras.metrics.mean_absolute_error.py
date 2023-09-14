@keras_export(
    "keras.metrics.mean_absolute_error",
    "keras.metrics.mae",
    "keras.metrics.MAE",
    "keras.losses.mean_absolute_error",
    "keras.losses.mae",
    "keras.losses.MAE",
)
@tf.__internal__.dispatch.add_dispatch_support
def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.
    `loss = mean(abs(y_true - y_pred), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return backend.mean(tf.abs(y_pred - y_true), axis=-1)
@dispatch.dispatch_for_types(mean_absolute_error, tf.RaggedTensor)
def _ragged_tensor_mae(y_true, y_pred):
    """RaggedTensor adapter for mean_absolute_error."""
    return _ragged_tensor_apply_loss(mean_absolute_error, y_true, y_pred)
