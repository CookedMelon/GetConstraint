"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.categorical_focal_crossentropy",
    "keras.losses.categorical_focal_crossentropy",
)
@tf.__internal__.dispatch.add_dispatch_support
def categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the categorical focal crossentropy loss.
    Standalone usage:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = tf.keras.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)
    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.
    Returns:
        Categorical focal crossentropy loss value.
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_focal_crossentropy, expected "
            "y_pred.shape to be (batch_size, num_classes) "
            f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
            "Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )
    def _smooth_labels():
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )
    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )
    return backend.categorical_focal_crossentropy(
        target=y_true,
        output=y_pred,
        alpha=alpha,
        gamma=gamma,
        from_logits=from_logits,
        axis=axis,
    )
@dispatch.dispatch_for_types(categorical_focal_crossentropy, tf.RaggedTensor)
def _ragged_tensor_categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Implements support for handling RaggedTensors.
    Expected shape: (batch, sequence_len, n_classes) with sequence_len
    being variable per batch.
    Return shape: (batch, sequence_len).
    When used by CategoricalFocalCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
    number of elements independent of the batch. E.g. if the RaggedTensor
    has 2 batches with [2, 1] values respectively the resulting loss is
    the sum of the individual loss values divided by 3.
    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.
    Returns:
      Categorical focal crossentropy loss value.
    """
    fn = functools.partial(
        categorical_focal_crossentropy,
        alpha=alpha,
        gamma=gamma,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis,
    )
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)
