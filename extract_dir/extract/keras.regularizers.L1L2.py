@keras_export("keras.regularizers.L1L2")
class L1L2(Regularizer):
    """A regularizer that applies both L1 and L2 regularization penalties.
    The L1 regularization penalty is computed as:
    `loss = l1 * reduce_sum(abs(x))`
    The L2 regularization penalty is computed as
    `loss = l2 * reduce_sum(square(x))`
    L1L2 may be passed to a layer as a string identifier:
    >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l1_l2')
    In this case, the default values used are `l1=0.01` and `l2=0.01`.
    Arguments:
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """
    def __init__(self, l1=0.0, l2=0.0):
        # The default value for l1 and l2 are different from the value in l1_l2
        # for backward compatibility reason. Eg, L1L2(l2=0.1) will only have l2
        # and no l1 penalty.
        l1 = 0.0 if l1 is None else l1
        l2 = 0.0 if l2 is None else l2
        _check_penalty_number(l1)
        _check_penalty_number(l2)
        self.l1 = backend.cast_to_floatx(l1)
        self.l2 = backend.cast_to_floatx(l2)
    def __call__(self, x):
        regularization = backend.constant(0.0, dtype=x.dtype)
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(x))
        if self.l2:
            # equivalent to "self.l2 * tf.reduce_sum(tf.square(x))"
            regularization += 2.0 * self.l2 * tf.nn.l2_loss(x)
        return regularization
    def get_config(self):
        return {"l1": float(self.l1), "l2": float(self.l2)}
