@keras_export("keras.metrics.RootMeanSquaredError")
class RootMeanSquaredError(base_metric.Mean):
    """Computes root mean squared error metric between `y_true` and `y_pred`.
    Standalone usage:
    >>> m = tf.keras.metrics.RootMeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.5
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.70710677
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="root_mean_squared_error", dtype=None):
        super().__init__(name, dtype=dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
        error_sq = tf.math.squared_difference(y_pred, y_true)
        return super().update_state(error_sq, sample_weight=sample_weight)
    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))
