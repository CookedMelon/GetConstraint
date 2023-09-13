@keras_export("keras.backend.learning_phase")
@doc_controls.do_not_generate_docs
def learning_phase():
    """Returns the learning phase flag.
    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.
    Returns:
        Learning phase (scalar integer tensor or Python integer).
    """
    graph = tf.compat.v1.get_default_graph()
    if graph is getattr(_GRAPH, "graph", None):
        # Don't enter an init_scope for the learning phase if eager execution
        # is enabled but we're inside the Keras workspace graph.
        learning_phase = symbolic_learning_phase()
    else:
        with tf.init_scope():
            # We always check & set the learning phase inside the init_scope,
            # otherwise the wrong default_graph will be used to look up the
            # learning phase inside of functions & defuns.
            #
            # This is because functions & defuns (both in graph & in eager mode)
            # will always execute non-eagerly using a function-specific default
            # subgraph.
            if context.executing_eagerly():
                if _DUMMY_EAGER_GRAPH.key not in _GRAPH_LEARNING_PHASES:
                    return _default_learning_phase()
                else:
                    return _internal_get_learning_phase(_DUMMY_EAGER_GRAPH.key)
            else:
                learning_phase = symbolic_learning_phase()
    _mark_func_graph_as_unsaveable(graph, learning_phase)
    return learning_phase
