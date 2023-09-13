@keras_export("keras.backend.map_fn")
@doc_controls.do_not_generate_docs
def map_fn(fn, elems, name=None, dtype=None):
    """Map the function fn over the elements elems and return the outputs.
    Args:
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph
        dtype: Output data type.
    Returns:
        Tensor with dtype `dtype`.
    """
    return tf.compat.v1.map_fn(fn, elems, name=name, dtype=dtype)
