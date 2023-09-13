@keras_export("keras.utils.pack_x_y_sample_weight", v1=[])
def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.
    This is a convenience utility for packing data into the tuple formats
    that `Model.fit` uses.
    Standalone usage:
    >>> x = tf.ones((10, 1))
    >>> data = tf.keras.utils.pack_x_y_sample_weight(x)
    >>> isinstance(data, tf.Tensor)
    True
    >>> y = tf.ones((10, 1))
    >>> data = tf.keras.utils.pack_x_y_sample_weight(x, y)
    >>> isinstance(data, tuple)
    True
    >>> x, y = data
    Args:
      x: Features to pass to `Model`.
      y: Ground-truth targets to pass to `Model`.
      sample_weight: Sample weight for each element.
    Returns:
      Tuple in the format used in `Model.fit`.
    """
    if y is None:
        # For single x-input, we do no tuple wrapping since in this case
        # there is no ambiguity. This also makes NumPy and Dataset
        # consistent in that the user does not have to wrap their Dataset
        # data in an unnecessary tuple.
        if not isinstance(x, tuple or list):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)
