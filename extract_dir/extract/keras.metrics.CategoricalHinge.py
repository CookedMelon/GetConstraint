@keras_export("keras.metrics.CategoricalHinge")
class CategoricalHinge(base_metric.MeanMetricWrapper):
    """Computes the categorical hinge metric between `y_true` and `y_pred`.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.CategoricalHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.4000001
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    1.2
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.CategoricalHinge()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="categorical_hinge", dtype=None):
        super().__init__(categorical_hinge, name, dtype=dtype)
