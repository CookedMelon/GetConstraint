@keras_export("keras.metrics.squared_hinge", "keras.losses.squared_hinge")
@tf.__internal__.dispatch.add_dispatch_support
def squared_hinge(y_true, y_pred):
    """Computes the squared hinge loss between `y_true` & `y_pred`.
    `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))
    Args:
      y_true: The ground truth values. `y_true` values are expected to be -1 or
        1. If binary (0 or 1) labels are provided we will convert them to -1 or
        1. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
       Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = _maybe_convert_labels(y_true)
    return backend.mean(
        tf.square(tf.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1
    )
