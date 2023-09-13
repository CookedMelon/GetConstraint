@keras_export("keras.regularizers.L2", "keras.regularizers.l2")
class L2(Regularizer):
    """A regularizer that applies a L2 regularization penalty.
    The L2 regularization penalty is computed as:
    `loss = l2 * reduce_sum(square(x))`
    L2 may be passed to a layer as a string identifier:
    >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l2')
    In this case, the default value used is `l2=0.01`.
    Arguments:
        l2: Float; L2 regularization factor.
    """
    def __init__(self, l2=0.01, **kwargs):
        l2 = kwargs.pop("l", l2)  # Backwards compatibility
        if kwargs:
            raise TypeError(f"Argument(s) not recognized: {kwargs}")
        l2 = 0.01 if l2 is None else l2
        _check_penalty_number(l2)
        self.l2 = backend.cast_to_floatx(l2)
    def __call__(self, x):
        # equivalent to "self.l2 * tf.reduce_sum(tf.square(x))"
        return 2.0 * self.l2 * tf.nn.l2_loss(x)
    def get_config(self):
        return {"l2": float(self.l2)}
