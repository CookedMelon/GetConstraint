@keras_export("keras.metrics.sparse_categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def sparse_categorical_accuracy(y_true, y_pred):
    """Calculates how often predictions match integer labels.
    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    Args:
      y_true: Integer ground truth values.
      y_pred: The prediction values.
    Returns:
      Sparse categorical accuracy values.
    """
    # Note: wraps metrics_utils.sparse_categorical_matches method and checks for
    # squeezing to align with expected public facing behavior. This seperates
    # public facing sparse_categorical_accuracy behavior from the vital behavior
    # of the sparse_categorical_matches method needed in backend dependencies.
    matches = metrics_utils.sparse_categorical_matches(y_true, y_pred)
    # if shape is (num_samples, 1) squeeze
    if matches.shape.ndims > 1 and matches.shape[-1] == 1:
        matches = tf.squeeze(matches, [-1])
    return matches
