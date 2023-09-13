"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.mean_squared_logarithmic_error",
    "keras.metrics.msle",
    "keras.metrics.MSLE",
    "keras.losses.mean_squared_logarithmic_error",
    "keras.losses.msle",
    "keras.losses.MSLE",
)
@tf.__internal__.dispatch.add_dispatch_support
def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.
    `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = np.maximum(y_true, 1e-7)
    >>> y_pred = np.maximum(y_pred, 1e-7)
    >>> assert np.allclose(
    ...     loss.numpy(),
    ...     np.mean(
    ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared logarithmic error values. shape = `[batch_size, d0, ..
      dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = tf.math.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    return backend.mean(
        tf.math.squared_difference(first_log, second_log), axis=-1
    )
@dispatch.dispatch_for_types(mean_squared_logarithmic_error, tf.RaggedTensor)
def _ragged_tensor_msle(y_true, y_pred):
    """Implements support for handling RaggedTensors."""
    return _ragged_tensor_apply_loss(
        mean_squared_logarithmic_error, y_true, y_pred
    )
def _maybe_convert_labels(y_true):
    """Converts binary labels into -1/1."""
    are_zeros = tf.equal(y_true, 0)
    are_ones = tf.equal(y_true, 1)
    is_binary = tf.reduce_all(tf.logical_or(are_zeros, are_ones))
    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2.0 * y_true - 1.0
    updated_y_true = tf.__internal__.smart_cond.smart_cond(
        is_binary, _convert_binary_labels, lambda: y_true
    )
    return updated_y_true
