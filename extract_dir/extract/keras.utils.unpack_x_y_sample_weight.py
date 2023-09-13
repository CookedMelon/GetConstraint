@keras_export("keras.utils.unpack_x_y_sample_weight", v1=[])
def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple.
    This is a convenience utility to be used when overriding
    `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
    This utility makes it easy to support data of the form `(x,)`,
    `(x, y)`, or `(x, y, sample_weight)`.
    Standalone usage:
    >>> features_batch = tf.ones((10, 5))
    >>> labels_batch = tf.zeros((10, 5))
    >>> data = (features_batch, labels_batch)
    >>> # `y` and `sample_weight` will default to `None` if not provided.
    >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
    >>> sample_weight is None
    True
    Example in overridden `Model.train_step`:
    ```python
    class MyModel(tf.keras.Model):
      def train_step(self, data):
        # If `sample_weight` is not provided, all samples will be weighted
        # equally.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
          y_pred = self(x, training=True)
          loss = self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
          trainable_variables = self.trainable_variables
          gradients = tape.gradient(loss, trainable_variables)
          self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
    ```
    Args:
      data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.
    Returns:
      The unpacked tuple, with `None`s for `y` and `sample_weight` if they are
      not provided.
    """
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    else:
        error_msg = (
            "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
            "or `(x, y, sample_weight)`, found: {}"
        ).format(data)
        raise ValueError(error_msg)
