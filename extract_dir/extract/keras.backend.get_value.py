@keras_export("keras.backend.get_value")
@doc_controls.do_not_generate_docs
def get_value(x):
    """Returns the value of a variable.
    `backend.get_value` is the complement of `backend.set_value`, and provides
    a generic interface for reading from variables while abstracting away the
    differences between TensorFlow 1.x and 2.x semantics.
    {snippet}
    Args:
        x: input variable.
    Returns:
        A Numpy array.
    """
    if not tf.is_tensor(x):
        return x
    if tf.executing_eagerly() or isinstance(x, tf.__internal__.EagerTensor):
        return x.numpy()
    if not getattr(x, "_in_graph_mode", True):
        # This is a variable which was created in an eager context, but is being
        # evaluated from a Graph.
        with tf.__internal__.eager_context.eager_mode():
            return x.numpy()
    if tf.compat.v1.executing_eagerly_outside_functions():
        # This method of evaluating works inside the Keras FuncGraph.
        with tf.init_scope():
            return x.numpy()
    with x.graph.as_default():
        return x.eval(session=get_session((x,)))
