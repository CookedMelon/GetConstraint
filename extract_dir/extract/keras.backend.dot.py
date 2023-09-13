@keras_export("keras.backend.dot")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a tensor.
    This operation corresponds to `numpy.dot(a, b, out=None)`.
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A tensor, dot product of `x` and `y`.
    Examples:
    If inputs `x` and `y` are 2-D arrays, then it is equivalent to `tf.matmul`.
    >>> x = tf.keras.backend.placeholder(shape=(2, 3))
    >>> y = tf.keras.backend.placeholder(shape=(3, 4))
    >>> xy = tf.keras.backend.dot(x, y)
    >>> xy
    <KerasTensor: shape=(2, 4) dtype=float32 ...>
    >>> x = tf.keras.backend.placeholder(shape=(32, 28, 3))
    >>> y = tf.keras.backend.placeholder(shape=(3, 4))
    >>> xy = tf.keras.backend.dot(x, y)
    >>> xy
    <KerasTensor: shape=(32, 28, 4) dtype=float32 ...>
    If `x` is an N-D array and `y` is an M-D array (where M>=2), it is a sum
    product over the last axis of `x` and the second-to-last axis of `y`.
    >>> x = tf.keras.backend.random_uniform_variable(
    ... shape=(2, 3), low=0., high=1.)
    >>> y = tf.keras.backend.ones((4, 3, 5))
    >>> xy = tf.keras.backend.dot(x, y)
    >>> tf.keras.backend.int_shape(xy)
    (2, 4, 5)
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(
            tf.compat.v1.transpose(y, perm=y_permute_dim), [y_shape[-2], -1]
        )
        return tf.reshape(
            tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:]
        )
    if is_sparse(x):
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out
