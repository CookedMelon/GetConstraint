@keras_export(
    "keras.metrics.categorical_crossentropy",
    "keras.losses.categorical_crossentropy",
)
@tf.__internal__.dispatch.add_dispatch_support
def categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the categorical crossentropy loss.
    Standalone usage:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)
    Args:
      y_true: Tensor of one-hot true targets.
      y_pred: Tensor of predicted targets.
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
        example, if `0.1`, use `0.1 / num_classes` for non-target labels
        and `0.9 + 0.1 / num_classes` for target labels.
      axis: Defaults to -1. The dimension along which the entropy is
        computed.
    Returns:
      Categorical crossentropy loss value.
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
            "In loss categorical_crossentropy, expected "
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
    return backend.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )
@dispatch.dispatch_for_types(categorical_crossentropy, tf.RaggedTensor)
def _ragged_tensor_categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Implements support for handling RaggedTensors.
    Args:
      y_true: Tensor of one-hot true targets.
      y_pred: Tensor of predicted targets.
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
        example, if `0.1`, use `0.1 / num_classes` for non-target labels
        and `0.9 + 0.1 / num_classes` for target labels.
      axis: The axis along which to compute crossentropy (the features axis).
          Defaults to -1.
    Returns:
      Categorical crossentropy loss value.
    Expected shape: (batch, sequence_len, n_classes) with sequence_len
    being variable per batch.
    Return shape: (batch, sequence_len).
    When used by CategoricalCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
    number of elements independent of the batch. E.g. if the RaggedTensor
    has 2 batches with [2, 1] values respectively the resulting loss is
    the sum of the individual loss values divided by 3.
    """
    fn = functools.partial(
        categorical_crossentropy,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis,
    )
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)
