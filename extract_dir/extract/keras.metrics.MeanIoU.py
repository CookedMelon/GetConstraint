@keras_export("keras.metrics.MeanIoU")
class MeanIoU(IoU):
    """Computes the mean Intersection-Over-Union metric.
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
    Note that this class first computes IoUs for all individual classes, then
    returns the mean of these values.
    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_true: Whether labels are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      sparse_y_pred: Whether predictions are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.
    Standalone usage:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        target_class_ids = list(range(num_classes))
        super().__init__(
            name=name,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            axis=axis,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
        )
    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
