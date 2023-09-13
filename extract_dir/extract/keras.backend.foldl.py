@keras_export("keras.backend.foldl")
@doc_controls.do_not_generate_docs
def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.
    Args:
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph
    Returns:
        Tensor with same type and shape as `initializer`.
    """
    return tf.compat.v1.foldl(fn, elems, initializer=initializer, name=name)
