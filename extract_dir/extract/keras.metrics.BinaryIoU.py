@keras_export("keras.metrics.BinaryIoU")
class BinaryIoU(IoU):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1.
    General definition and computation:
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.
    For an individual class, the IoU metric is defined as follows:
    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    This class can be used to compute IoUs for a binary classification task
    where the predictions are provided as logits. First a `threshold` is applied
    to the predicted values such that those that are below the `threshold` are
    converted to class 0 and those that are above the `threshold` are converted
    to class 1.
    IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
    that are specified by `target_class_ids` is returned.
    Note: with `threshold=0`, this metric has the same behavior as `IoU`.
    Args:
      target_class_ids: A tuple or list of target class ids for which the metric
        is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With `[0]` (or
        `[1]`), the IoU metric for class 0 (or class 1, respectively) is
        returned. With `[0, 1]`, the mean of IoUs for the two classes is
        returned.
      threshold: A threshold that applies to the prediction logits to convert
        them to either predicted class 0 if the logit is below `threshold` or
        predicted class 1 if the logit is above `threshold`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> m = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
    ...                sample_weight=[0.2, 0.3, 0.4, 0.1])
    >>> # cm = [[0.2, 0.4],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
    >>> # true_positives = [0.2, 0.1]
    >>> # iou = [0.222, 0.125]
    >>> m.result().numpy()
    0.17361112
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        target_class_ids: Union[List[int], Tuple[int, ...]] = (0, 1),
        threshold=0.5,
        name=None,
        dtype=None,
    ):
        super().__init__(
            num_classes=2,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
        )
        self.threshold = threshold
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Before the confusion matrix is updated, the predicted values are
        thresholded to be:
          0 for values that are smaller than the `threshold`
          1 for values that are larger or equal to the `threshold`
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred = tf.cast(y_pred >= self.threshold, self._dtype)
        return super().update_state(y_true, y_pred, sample_weight)
    def get_config(self):
        return {
            "target_class_ids": self.target_class_ids,
            "threshold": self.threshold,
            "name": self.name,
            "dtype": self._dtype,
        }
