@keras_export("keras.regularizers.L1", "keras.regularizers.l1")
class L1(Regularizer):
    """A regularizer that applies a L1 regularization penalty.
    The L1 regularization penalty is computed as:
    `loss = l1 * reduce_sum(abs(x))`
    L1 may be passed to a layer as a string identifier:
    >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='l1')
    In this case, the default value used is `l1=0.01`.
    Arguments:
        l1: Float; L1 regularization factor.
    """
    def __init__(self, l1=0.01, **kwargs):
        l1 = kwargs.pop("l", l1)  # Backwards compatibility
        if kwargs:
            raise TypeError(f"Argument(s) not recognized: {kwargs}")
        l1 = 0.01 if l1 is None else l1
        _check_penalty_number(l1)
        self.l1 = backend.cast_to_floatx(l1)
    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x))
    def get_config(self):
        return {"l1": float(self.l1)}
