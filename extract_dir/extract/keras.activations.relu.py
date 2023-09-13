@keras_export("keras.activations.relu")
@tf.__internal__.dispatch.add_dispatch_support
def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    """Applies the rectified linear unit activation function.
    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.
    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation,
    and to use a non-zero multiple of the input for values below the threshold.
    Example:
    >>> foo = tf.constant([-10, -5, 0.0, 5, 10], dtype = tf.float32)
    >>> tf.keras.activations.relu(foo).numpy()
    array([ 0.,  0.,  0.,  5., 10.], dtype=float32)
    >>> tf.keras.activations.relu(foo, alpha=0.5).numpy()
    array([-5. , -2.5,  0. ,  5. , 10. ], dtype=float32)
    >>> tf.keras.activations.relu(foo, max_value=5.).numpy()
    array([0., 0., 0., 5., 5.], dtype=float32)
    >>> tf.keras.activations.relu(foo, threshold=5.).numpy()
    array([-0., -0.,  0.,  0., 10.], dtype=float32)
    Args:
        x: Input `tensor` or `variable`.
        alpha: A `float` that governs the slope for values lower than the
          threshold.
        max_value: A `float` that sets the saturation threshold (the largest
          value the function will return).
        threshold: A `float` giving the threshold value of the activation
          function below which values will be damped or set to zero.
    Returns:
        A `Tensor` representing the input tensor,
        transformed by the relu activation function.
        Tensor will be of the same shape and dtype of input `x`.
    """
    return backend.relu(
        x, alpha=alpha, max_value=max_value, threshold=threshold
    )
