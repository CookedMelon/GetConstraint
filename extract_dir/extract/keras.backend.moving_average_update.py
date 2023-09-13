@keras_export("keras.backend.moving_average_update")
@doc_controls.do_not_generate_docs
def moving_average_update(x, value, momentum):
    """Compute the exponential moving average of a value.
    The moving average 'x' is updated with 'value' following:
    ```
    x = x * momentum + value * (1 - momentum)
    ```
    For example:
    >>> x = tf.Variable(0.0)
    >>> momentum=0.9
    >>> moving_average_update(x, value = 2.0, momentum=momentum).numpy()
    >>> x.numpy()
    0.2
    The result will be biased towards the initial value of the variable.
    If the variable was initialized to zero, you can divide by
    `1 - momentum ** num_updates` to debias it (Section 3 of
    [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)):
    >>> num_updates = 1.0
    >>> x_zdb = x/(1 - momentum**num_updates)
    >>> x_zdb.numpy()
    2.0
    Args:
        x: A Variable, the moving average.
        value: A tensor with the same shape as `x`, the new value to be
          averaged in.
        momentum: The moving average momentum.
    Returns:
        The updated variable.
    """
    if tf.__internal__.tf2.enabled():
        momentum = tf.cast(momentum, x.dtype)
        value = tf.cast(value, x.dtype)
        return x.assign_sub((x - value) * (1 - momentum))
    else:
        return tf.__internal__.train.assign_moving_average(
            x, value, momentum, zero_debias=True
        )
