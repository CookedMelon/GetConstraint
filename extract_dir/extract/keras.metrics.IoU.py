@keras_export("keras.metrics.IoU")
class IoU(_IoUBase):
    """Computes the Intersection-Over-Union metric for specific target classes.
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
    Note, this class first computes IoUs for all individual classes, then
    returns the mean of IoUs for the classes that are specified by
    `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
    that specific class is returned.
    Args:
      num_classes: The possible number of labels the prediction task can have.
        A confusion matrix of dimension = [num_classes, num_classes] will be
        allocated to accumulate predictions from which the metric is calculated.
      target_class_ids: A tuple or list of target class ids for which the metric
        is returned. To compute IoU for a specific class, a list (or tuple) of a
        single id value should be provided.
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
    >>> # iou = [0.33, 0.33]
    >>> m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> # cm = [[0.3, 0.3],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4],
    >>> # true_positives = [0.3, 0.1]
    >>> # iou = [0.33, 0.14]
    >>> m.result().numpy()
    0.33333334
    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
    ```
    """
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        target_class_ids: Union[List[int], Tuple[int, ...]],
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            dtype=dtype,
        )
        if max(target_class_ids) >= num_classes:
            raise ValueError(
                f"Target class id {max(target_class_ids)} "
                "is out of range, which is "
                f"[{0}, {num_classes})."
            )
        self.target_class_ids = list(target_class_ids)
    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )
        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives
        # Only keep the target classes
        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)
        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )
        iou = tf.math.divide_no_nan(true_positives, denominator)
        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name="mean_iou"), num_valid_entries
        )
    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
