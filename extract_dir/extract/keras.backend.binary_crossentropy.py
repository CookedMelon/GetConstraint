@keras_export("keras.backend.binary_crossentropy")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    Args:
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    output, from_logits = _get_logits(
        output, from_logits, "Sigmoid", "binary_crossentropy"
    )
    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    # Compute cross entropy from probabilities.
    bce = target * tf.math.log(output + epsilon())
    bce += (1 - target) * tf.math.log(1 - output + epsilon())
    return -bce
