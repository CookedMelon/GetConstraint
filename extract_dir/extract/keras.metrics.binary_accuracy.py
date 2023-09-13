@keras_export("keras.metrics.binary_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def binary_accuracy(y_true, y_pred, threshold=0.5):
    """Calculates how often predictions match binary labels.
    Standalone usage:
    >>> y_true = [[1], [1], [0], [0]]
    >>> y_pred = [[1], [1], [0], [0]]
    >>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
    >>> assert m.shape == (4,)
    >>> m.numpy()
    array([1., 1., 1., 1.], dtype=float32)
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.
    Returns:
      Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
    """
    # Note: calls metrics_utils.binary_matches with mean reduction. This
    # maintains public facing binary_accuracy behavior and seperates it from the
    # vital behavior of the binary_matches method needed in backend
    # dependencies.
    return tf.reduce_mean(
        metrics_utils.binary_matches(y_true, y_pred, threshold), axis=-1
    )
