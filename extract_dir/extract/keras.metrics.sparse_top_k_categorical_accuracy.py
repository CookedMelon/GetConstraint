@keras_export("keras.metrics.sparse_top_k_categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Computes how often integer targets are in the top `K` predictions.
    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
    ...     y_true, y_pred, k=3)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([1., 1.], dtype=float32)
    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
    Returns:
      Sparse top K categorical accuracy value.
    """
    # Note: wraps metrics_utils.sparse_top_k_categorical_matches. This seperates
    # public facing sparse_top_k_categorical_accuracy behavior from the vital
    # behavior of the sparse_top_k_categorical_matches method needed in backend
    # dependencies.
    return metrics_utils.sparse_top_k_categorical_matches(y_true, y_pred, k)
