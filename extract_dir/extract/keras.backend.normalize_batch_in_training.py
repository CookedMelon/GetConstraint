@keras_export("keras.backend.normalize_batch_in_training")
@doc_controls.do_not_generate_docs
def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.
    Args:
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.
    Returns:
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    if ndim(x) == 4 and list(reduction_axes) in [[0, 1, 2], [0, 2, 3]]:
        if not _has_nchw_support() and list(reduction_axes) == [0, 2, 3]:
            return _broadcast_normalize_batch_in_training(
                x, gamma, beta, reduction_axes, epsilon=epsilon
            )
        return _fused_normalize_batch_in_training(
            x, gamma, beta, reduction_axes, epsilon=epsilon
        )
    else:
        if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
            return _regular_normalize_batch_in_training(
                x, gamma, beta, reduction_axes, epsilon=epsilon
            )
        else:
            return _broadcast_normalize_batch_in_training(
                x, gamma, beta, reduction_axes, epsilon=epsilon
            )
