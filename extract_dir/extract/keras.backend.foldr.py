@keras_export("keras.backend.foldr")
@doc_controls.do_not_generate_docs
def foldr(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from right to left.
    Args:
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[-1]` in case of None)
        name: A string name for the foldr node in the graph
    Returns:
        Same type and shape as initializer
    """
    return tf.compat.v1.foldr(fn, elems, initializer=initializer, name=name)
