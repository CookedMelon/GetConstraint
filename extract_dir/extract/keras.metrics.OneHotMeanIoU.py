@keras_export("keras.metrics.OneHotMeanIoU")
class OneHotMeanIoU(MeanIoU):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels.
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
    This class can be used to compute the mean IoU for multi-class
    classification tasks where the labels are one-hot encoded (the last axis
    should have one dimension per class). Note that the predictions should also
    have the same shape. To compute the mean IoU, first the labels and
    predictions are converted back into integer format by taking the argmax over
    the class axis. Then the same computation steps as for the base `MeanIoU`
    class apply.
    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.
    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.
    Args:
      num_classes: The possible number of labels the prediction task can have.
        A confusion matrix of shape `(num_classes, num_classes)` will be
        allocated to accumulate predictions from which the metric is calculated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_pred: Whether predictions are encoded using natural numbers or
        probability distribution vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.
    Standalone usage:
    >>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = tf.keras.metrics.OneHotMeanIoU(num_classes=3)
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
    >>> m.result().numpy()
    0.048
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=3)])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        name: str = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = False,
        axis: int = -1,
    ):
        super().__init__(
            num_classes=num_classes,
            axis=axis,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
        )
    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
