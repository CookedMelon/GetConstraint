@keras_export("keras.metrics.F1Score")
class F1Score(FBetaScore):
    r"""Computes F-1 Score.
    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.
    It is defined as:
    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```
    Args:
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `"micro"`, `"macro"`
            and `"weighted"`. Default value is `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.
    Returns:
        F-1 Score: float.
    Example:
    >>> metric = tf.keras.metrics.F1Score(threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.5      , 0.8      , 0.6666667], dtype=float32)
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )
    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
