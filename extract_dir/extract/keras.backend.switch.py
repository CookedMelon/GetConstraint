@keras_export("keras.backend.switch")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value.
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.
    Args:
        condition: tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.
    Returns:
        The selected tensor.
    Raises:
        ValueError: If rank of `condition` is greater than rank of expressions.
    """
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, "bool")
    cond_ndim = ndim(condition)
    if not cond_ndim:
        if not callable(then_expression):
            def then_expression_fn():
                return then_expression
        else:
            then_expression_fn = then_expression
        if not callable(else_expression):
            def else_expression_fn():
                return else_expression
        else:
            else_expression_fn = else_expression
        x = tf.compat.v1.cond(condition, then_expression_fn, else_expression_fn)
    else:
        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        expr_ndim = ndim(then_expression)
        if cond_ndim > expr_ndim:
            raise ValueError(
                "Rank of `condition` should be less than or"
                " equal to rank of `then_expression` and "
                "`else_expression`. ndim(condition)="
                + str(cond_ndim)
                + ", ndim(then_expression)="
                + str(expr_ndim)
            )
        if cond_ndim > 1:
            ndim_diff = expr_ndim - cond_ndim
            cond_shape = tf.concat(
                [tf.shape(condition), [1] * ndim_diff], axis=0
            )
            condition = tf.reshape(condition, cond_shape)
            expr_shape = tf.shape(then_expression)
            shape_diff = expr_shape - cond_shape
            tile_shape = tf.where(
                shape_diff > 0, expr_shape, tf.ones_like(expr_shape)
            )
            condition = tf.tile(condition, tile_shape)
        x = tf.where(condition, then_expression, else_expression)
    return x
