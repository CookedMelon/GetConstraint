@keras_export("keras.backend.in_test_phase")
@doc_controls.do_not_generate_docs
def in_test_phase(x, alt, training=None):
    """Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    Args:
        x: What to return in test phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.
    Returns:
        Either `x` or `alt` based on `K.learning_phase`.
    """
    return in_train_phase(alt, x, training=training)
