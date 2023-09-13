@keras_export("keras.backend.categorical_crossentropy")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.
    Args:
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.
    Returns:
        Output tensor.
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    Example:
    >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0.]
       [0. 1. 0.]
       [0. 0. 1.]], shape=(3, 3), dtype=float32)
    >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94],
    ...                 shape=[3, 3])
    >>> print(b)
    tf.Tensor(
      [[0.9  0.05 0.05]
       [0.05 0.89 0.06]
       [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.11653 0.06188]
    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0.]
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)
    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "categorical_crossentropy"
    )
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )
    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, True)
    # Compute cross entropy from probabilities.
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target * tf.math.log(output), axis)
