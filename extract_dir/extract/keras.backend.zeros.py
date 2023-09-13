@keras_export("keras.backend.zeros")
@doc_controls.do_not_generate_docs
def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.
    Args:
        shape: Tuple or list of integers, shape of returned Keras variable
        dtype: data type of returned Keras variable
        name: name of returned Keras variable
    Returns:
        A variable (including Keras metadata), filled with `0.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.
    Example:
    >>> kvar = tf.keras.backend.zeros((3,4))
    >>> tf.keras.backend.eval(kvar)
    array([[0.,  0.,  0.,  0.],
           [0.,  0.,  0.,  0.],
           [0.,  0.,  0.,  0.]], dtype=float32)
    >>> A = tf.constant([1,2,3])
    >>> kvar2 = tf.keras.backend.zeros(A.shape) # [0., 0., 0.]
    >>> tf.keras.backend.eval(kvar2)
    array([0., 0., 0.], dtype=float32)
    >>> kvar3 = tf.keras.backend.zeros(A.shape,dtype=tf.int32)
    >>> tf.keras.backend.eval(kvar3)
    array([0, 0, 0], dtype=int32)
    >>> kvar4 = tf.keras.backend.zeros([2,3])
    >>> tf.keras.backend.eval(kvar4)
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    """
    with tf.init_scope():
        if dtype is None:
            dtype = floatx()
        tf_dtype = tf.as_dtype(dtype)
        v = tf.zeros(shape=shape, dtype=tf_dtype, name=name)
        if py_all(v.shape.as_list()):
            return variable(v, dtype=dtype, name=name)
        return v
