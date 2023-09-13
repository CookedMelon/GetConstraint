"/home/cc/Workspace/tfconstraint/keras/losses.py"
@keras_export(
    "keras.metrics.binary_crossentropy", "keras.losses.binary_crossentropy"
)
@tf.__internal__.dispatch.add_dispatch_support
def binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the binary crossentropy loss.
    Standalone usage:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels by
        squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
        for the target class and `0.5 * label_smoothing` for the non-target
        class.
      axis: The axis along which the mean is computed. Defaults to -1.
    Returns:
      Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )
    return backend.mean(
        backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    )
@dispatch.dispatch_for_types(binary_crossentropy, tf.RaggedTensor)
def _ragged_tensor_binary_crossentropy(
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
      axis: Axis along which to compute crossentropy.
    Returns:
      Binary crossentropy loss value.
    Expected shape: (batch, sequence_len) with sequence_len being variable
    per batch.
    Return shape: (batch,); returns the per batch mean of the loss values.
    When used by BinaryCrossentropy() with the default reduction
    (SUM_OVER_BATCH_SIZE), the reduction averages the per batch losses over
    the number of batches.
    """
    fn = functools.partial(
        binary_crossentropy,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis,
    )
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)
