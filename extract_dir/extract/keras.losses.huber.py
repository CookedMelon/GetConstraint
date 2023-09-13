@keras_export("keras.losses.huber", v1=[])
@tf.__internal__.dispatch.add_dispatch_support
def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.
    For each value x in `error = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = d * |x| - 0.5 * d^2        if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
    Returns:
      Tensor with one scalar loss entry per sample.
    """
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    delta = tf.cast(delta, dtype=backend.floatx())
    error = tf.subtract(y_pred, y_true)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return backend.mean(
        tf.where(
            abs_error <= delta,
            half * tf.square(error),
            delta * abs_error - half * tf.square(delta),
        ),
        axis=-1,
    )
