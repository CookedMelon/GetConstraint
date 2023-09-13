@keras_export("keras.layers.Maximum")
class Maximum(_Merge):
    """Layer that computes the maximum (element-wise) a list of inputs.
    It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).
    >>> tf.keras.layers.Maximum()([np.arange(5).reshape(5, 1),
    ...                            np.arange(5, 10).reshape(5, 1)])
    <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[5],
         [6],
         [7],
         [8],
         [9]])>
    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> maxed = tf.keras.layers.Maximum()([x1, x2])
    >>> maxed.shape
    TensorShape([5, 8])
    """
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = tf.maximum(output, inputs[i])
        return output
