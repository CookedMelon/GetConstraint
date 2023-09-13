@keras_export("keras.metrics.categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def categorical_accuracy(y_true, y_pred):
    """Calculates how often predictions match one-hot labels.
    Standalone usage:
    >>> y_true = [[0, 0, 1], [0, 1, 0]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    Args:
      y_true: One-hot ground truth values.
      y_pred: The prediction values.
    Returns:
      Categorical accuracy values.
    """
    # Note: wraps metrics_utils.categorical_matches. This seperates public
    # facing categorical_accuracy behavior from the vital behavior of the
    # categorical_matches method needed in backend dependencies.
    return metrics_utils.sparse_categorical_matches(
        tf.math.argmax(y_true, axis=-1), y_pred
    )
