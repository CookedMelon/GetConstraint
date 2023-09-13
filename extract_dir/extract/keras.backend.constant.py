@keras_export("keras.backend.constant")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def constant(value, dtype=None, shape=None, name=None):
    """Creates a constant tensor.
    Args:
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.
    Returns:
        A Constant Tensor.
    """
    if dtype is None:
        dtype = floatx()
    return tf.constant(value, dtype=dtype, shape=shape, name=name)
