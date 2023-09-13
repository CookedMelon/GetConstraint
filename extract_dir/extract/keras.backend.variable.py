@keras_export("keras.backend.variable")
@doc_controls.do_not_generate_docs
def variable(value, dtype=None, name=None, constraint=None):
    """Instantiates a variable and returns it.
    Args:
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.
    Returns:
        A variable instance (with Keras metadata included).
    Examples:
    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = tf.keras.backend.variable(value=val, dtype='float64',
    ...                                  name='example_var')
    >>> tf.keras.backend.dtype(kvar)
    'float64'
    >>> print(kvar)
    <tf.Variable 'example_var:...' shape=(2, 2) dtype=float64, numpy=
      array([[1., 2.],
             [3., 4.]])>
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, "tocoo"):
        sparse_coo = value.tocoo()
        indices = np.concatenate(
            (
                np.expand_dims(sparse_coo.row, 1),
                np.expand_dims(sparse_coo.col, 1),
            ),
            1,
        )
        v = tf.SparseTensor(
            indices=indices,
            values=sparse_coo.data,
            dense_shape=sparse_coo.shape,
        )
        v._keras_shape = sparse_coo.shape
        return v
    v = tf.Variable(
        value, dtype=tf.as_dtype(dtype), name=name, constraint=constraint
    )
    if isinstance(value, np.ndarray):
        v._keras_shape = value.shape
    elif hasattr(value, "shape"):
        v._keras_shape = int_shape(value)
    track_variable(v)
    return v
