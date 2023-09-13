@keras_export("keras.losses.CategoricalFocalCrossentropy")
class CategoricalFocalCrossentropy(LossFunctionWrapper):
    """Computes the alpha balanced focal crossentropy loss.
    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    `class_weights`. We expect labels to be provided in a `one_hot`
    representation.
    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:
    `FL(p_t) = (1 − p_t)^gamma * log(p_t)`
    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`
    `(1 − p_t)^gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.
    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = −alpha * (1 − p_t)^gamma * log(p_t)`
    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.
    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CrossEntropy(y_true, y_pred)`
    where minus comes from `CrossEntropy(y_true, y_pred)` (CE).
    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CategoricalCE(y_true, y_pred)`
    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.
    Standalone usage:
    >>> y_true = [[0., 1., 0.], [0., 0., 1.]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy()
    >>> cce(y_true, y_pred).numpy()
    0.23315276
    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.1632
    >>> # Using 'sum' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cce(y_true, y_pred).numpy()
    0.46631
    >>> # Using 'none' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cce(y_true, y_pred).numpy()
    array([3.2058331e-05, 4.6627346e-01], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalFocalCrossentropy())
    ```
    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to -1.
        reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
        name: Optional name for the instance.
            Defaults to 'categorical_focal_crossentropy'.
    """
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="categorical_focal_crossentropy",
    ):
        """Initializes `CategoricalFocalCrossentropy` instance."""
        super().__init__(
            categorical_focal_crossentropy,
            alpha=alpha,
            gamma=gamma,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
    def get_config(self):
        config = {
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
