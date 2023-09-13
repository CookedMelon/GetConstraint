@keras_export("keras.layers.Multiply")
class Multiply(_Merge):
    """Layer that multiplies (element-wise) a list of inputs.
    It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).
    >>> tf.keras.layers.Multiply()([np.arange(5).reshape(5, 1),
    ...                             np.arange(5, 10).reshape(5, 1)])
    <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[ 0],
         [ 6],
         [14],
         [24],
         [36]])>
    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> multiplied = tf.keras.layers.Multiply()([x1, x2])
    >>> multiplied.shape
    TensorShape([5, 8])
    """
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output * inputs[i]
        return output
