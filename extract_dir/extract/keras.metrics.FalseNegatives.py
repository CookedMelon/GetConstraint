@keras_export("keras.metrics.FalseNegatives")
class FalseNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of false negatives.
    If `sample_weight` is given, calculates the sum of the weights of
    false negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false negatives.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.FalseNegatives()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
    >>> m.result().numpy()
    2.0
    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.FalseNegatives()])
    ```
    Usage with a loss with `from_logits=True`:
    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.FalseNegatives(thresholds=0)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )
