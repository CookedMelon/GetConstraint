@keras_export("keras.metrics.Hinge")
class Hinge(base_metric.MeanMetricWrapper):
    """Computes the hinge metric between `y_true` and `y_pred`.
    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.Hinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.3
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    1.1
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.Hinge()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="hinge", dtype=None):
        super().__init__(hinge, name, dtype=dtype)
