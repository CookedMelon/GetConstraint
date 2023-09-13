@keras_export("keras.metrics.Precision")
class Precision(base_metric.Metric):
    """Computes the precision of the predictions with respect to the labels.
    The metric creates two local variables, `true_positives` and
    `false_positives` that are used to compute the precision. This value is
    ultimately returned as `precision`, an idempotent operation that simply
    divides `true_positives` by the sum of `true_positives` and
    `false_positives`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry
    is correct and can be found in the label for that entry.
    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in
    the top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label.
    Args:
      thresholds: (Optional) A float value, or a Python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). If used with a loss function
        that sets `from_logits=True` (i.e. no sigmoid applied to predictions),
        `thresholds` should be set to 0. One metric value is generated for each
        threshold value. If neither thresholds nor top_k are set, the default is
        to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating precision.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.Precision()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result().numpy()
    0.6666667
    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0
    >>> # With top_k=2, it will calculate precision over y_true[:2]
    >>> # and y_pred[:2]
    >>> m = tf.keras.metrics.Precision(top_k=2)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result().numpy()
    0.0
    >>> # With top_k=4, it will calculate precision over y_true[:4]
    >>> # and y_pred[:4]
    >>> m = tf.keras.metrics.Precision(top_k=4)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result().numpy()
    0.5
    Usage with `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Precision()])
    ```
    Usage with a loss with `from_logits=True`:
    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Precision(thresholds=0)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=(len(self.thresholds),),
            initializer="zeros",
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false positive statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )
    def result(self):
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        return result[0] if len(self.thresholds) == 1 else result
    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_positives, self.false_positives)
            ]
        )
    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
