@keras_export("keras.backend.batch_normalization")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.
    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
    Args:
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.
    Returns:
        A tensor.
    """
    if ndim(x) == 4:
        # The CPU implementation of `fused_batch_norm` only supports NHWC
        if axis == 1 or axis == -3:
            tf_data_format = "NCHW"
        elif axis == 3 or axis == -1:
            tf_data_format = "NHWC"
        else:
            tf_data_format = None
        if (
            tf_data_format == "NHWC"
            or tf_data_format == "NCHW"
            and _has_nchw_support()
        ):
            # The mean / var / beta / gamma tensors may be broadcasted
            # so they may have extra axes of size 1, which should be squeezed.
            if ndim(mean) > 1:
                mean = tf.reshape(mean, [-1])
            if ndim(var) > 1:
                var = tf.reshape(var, [-1])
            if beta is None:
                beta = zeros_like(mean)
            elif ndim(beta) > 1:
                beta = tf.reshape(beta, [-1])
            if gamma is None:
                gamma = ones_like(mean)
            elif ndim(gamma) > 1:
                gamma = tf.reshape(gamma, [-1])
        y, _, _ = tf.compat.v1.nn.fused_batch_norm(
            x,
            gamma,
            beta,
            epsilon=epsilon,
            mean=mean,
            variance=var,
            data_format=tf_data_format,
            is_training=False,
        )
        return y
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
