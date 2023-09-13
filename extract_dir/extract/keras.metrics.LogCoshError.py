@keras_export("keras.metrics.LogCoshError")
class LogCoshError(base_metric.MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.
    `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred -
    y_true)
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.LogCoshError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.10844523
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.21689045
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.LogCoshError()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="logcosh", dtype=None):
        super().__init__(logcosh, name, dtype=dtype)
