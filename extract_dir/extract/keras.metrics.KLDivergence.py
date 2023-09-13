@keras_export("keras.metrics.KLDivergence")
class KLDivergence(base_metric.MeanMetricWrapper):
    """Computes Kullback-Leibler divergence metric between `y_true` and
    `y_pred`.
    `metric = y_true * log(y_true / y_pred)`
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.KLDivergence()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    0.45814306
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.9162892
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.KLDivergence()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="kullback_leibler_divergence", dtype=None):
        super().__init__(kullback_leibler_divergence, name, dtype=dtype)
