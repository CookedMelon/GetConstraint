@keras_export("keras.layers.concatenate")
def concatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `Concatenate` layer.
    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> print(x)
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
     [[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> print(y)
    [[[20 21 22 23 24]]
     [[25 26 27 28 29]]]
    >>> tf.keras.layers.concatenate([x, y],
    ...                             axis=1)
    <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [20, 21, 22, 23, 24]],
         [[10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [25, 26, 27, 28, 29]]])>
    Args:
        inputs: A list of input tensors.
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    return Concatenate(axis=axis, **kwargs)(inputs)
