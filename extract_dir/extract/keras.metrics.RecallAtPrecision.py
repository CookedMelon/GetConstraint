@keras_export("keras.metrics.RecallAtPrecision")
class RecallAtPrecision(SensitivitySpecificityBase):
    """Computes best recall where precision is >= specified value.
    For a given score-label-distribution the required precision might not
    be achievable, in this case 0.0 is returned as recall.
    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the recall at the given precision. The threshold for the given
    precision value is computed and used to evaluate the corresponding recall.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.
    Args:
      precision: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use for matching the given precision.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.RecallAtPrecision(0.8)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> m.result().numpy()
    0.5
    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    1.0
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        precision,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
    ):
        if precision < 0 or precision > 1:
            raise ValueError(
                "Argument `precision` must be in the range [0, 1]. "
                f"Received: precision={precision}"
            )
        self.precision = precision
        self.num_thresholds = num_thresholds
        super().__init__(
            value=precision,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
    def result(self):
        precisions = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        recalls = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        return self._find_max_under_constraint(
            precisions, recalls, tf.greater_equal
        )
    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "precision": self.precision,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
