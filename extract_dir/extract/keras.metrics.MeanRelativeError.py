@keras_export("keras.metrics.MeanRelativeError")
class MeanRelativeError(base_metric.Mean):
    """Computes the mean relative error by normalizing with the given values.
    This metric creates two local variables, `total` and `count` that are used
    to compute the mean relative error. This is weighted by `sample_weight`, and
    it is ultimately returned as `mean_relative_error`: an idempotent operation
    that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      normalizer: The normalizer values with same shape as predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
    >>> m.update_state([1, 3, 2, 3], [2, 4, 6, 8])
    >>> # metric = mean(|y_pred - y_true| / normalizer)
    >>> #        = mean([1, 1, 4, 5] / [1, 3, 2, 3]) = mean([1, 1/3, 2, 5/3])
    >>> #        = 5/4 = 1.25
    >>> m.result().numpy()
    1.25
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanRelativeError(normalizer=[1, 3])])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, normalizer, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        normalizer = tf.cast(normalizer, self._dtype)
        self.normalizer = normalizer
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.
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
        [
            y_pred,
            y_true,
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(  # noqa: E501
            [y_pred, y_true], sample_weight
        )
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
        y_pred, self.normalizer = losses_utils.remove_squeezable_dimensions(
            y_pred, self.normalizer
        )
        y_pred.shape.assert_is_compatible_with(y_true.shape)
        relative_errors = tf.math.divide_no_nan(
            tf.abs(y_true - y_pred), self.normalizer
        )
        return super().update_state(
            relative_errors, sample_weight=sample_weight
        )
    def get_config(self):
        n = self.normalizer
        config = {
            "normalizer": backend.eval(n) if is_tensor_or_variable(n) else n
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
