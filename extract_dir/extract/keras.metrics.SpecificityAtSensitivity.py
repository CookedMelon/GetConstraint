@keras_export("keras.metrics.SpecificityAtSensitivity")
class SpecificityAtSensitivity(SensitivitySpecificityBase):
    """Computes best specificity where sensitivity is >= specified value.
    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such (tp / (tp + fn)).
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such (tn / (tn + fp)).
    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the specificity at the given sensitivity. The threshold for the
    given sensitivity value is computed and used to evaluate the corresponding
    specificity.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.
    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
    Args:
      sensitivity: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use for matching the given sensitivity.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.SpecificityAtSensitivity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result().numpy()
    0.66666667
    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 2])
    >>> m.result().numpy()
    0.5
    Usage with `compile()` API:
    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SpecificityAtSensitivity()])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        sensitivity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
    ):
        if sensitivity < 0 or sensitivity > 1:
            raise ValueError(
                "Argument `sensitivity` must be in the range [0, 1]. "
                f"Received: sensitivity={sensitivity}"
            )
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        super().__init__(
            sensitivity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
    def result(self):
        sensitivities = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        specificities = tf.math.divide_no_nan(
            self.true_negatives,
            tf.math.add(self.true_negatives, self.false_positives),
        )
        return self._find_max_under_constraint(
            sensitivities, specificities, tf.greater_equal
        )
    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "sensitivity": self.sensitivity,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
