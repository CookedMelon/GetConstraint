@keras_export("keras.backend.arange")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def arange(start, stop=None, step=1, dtype="int32"):
    """Creates a 1D tensor containing a sequence of integers.
    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument and "start" is 0.
    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.
    Args:
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.
    Returns:
        An integer tensor.
    Example:
        >>> tf.keras.backend.arange(start=0, stop=10, step=1.5)
        <tf.Tensor: shape=(7,), dtype=float32,
            numpy=array([0. , 1.5, 3. , 4.5, 6. , 7.5, 9. ], dtype=float32)>
    """
    # Match the behavior of numpy and Theano by returning an empty sequence.
    if stop is None and start < 0:
        start = 0
    result = tf.range(start, limit=stop, delta=step, name="arange")
    if dtype != "int32":
        result = cast(result, dtype)
    return result
