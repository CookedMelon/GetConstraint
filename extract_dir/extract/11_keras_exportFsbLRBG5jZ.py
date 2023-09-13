"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.mean_absolute_percentage_error",
    "keras.metrics.mape",
    "keras.metrics.MAPE",
    "keras.losses.mean_absolute_percentage_error",
    "keras.losses.mape",
    "keras.losses.MAPE",
)
@tf.__internal__.dispatch.add_dispatch_support
def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.
    `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.random(size=(2, 3))
    >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean absolute percentage error values. shape = `[batch_size, d0, ..
      dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.abs(
        (y_true - y_pred) / backend.maximum(tf.abs(y_true), backend.epsilon())
    )
    return 100.0 * backend.mean(diff, axis=-1)
@dispatch.dispatch_for_types(mean_absolute_percentage_error, tf.RaggedTensor)
def _ragged_tensor_mape(y_true, y_pred):
    """Support RaggedTensors."""
    return _ragged_tensor_apply_loss(
        mean_absolute_percentage_error, y_true, y_pred
    )
