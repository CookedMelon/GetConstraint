@keras_export("keras.backend.learning_phase_scope")
@tf_contextlib.contextmanager
@doc_controls.do_not_generate_docs
def learning_phase_scope(value):
    """Provides a scope within which the learning phase is equal to `value`.
    The learning phase gets restored to its original value upon exiting the
    scope.
    Args:
       value: Learning phase value, either 0 or 1 (integers).
              0 = test, 1 = train
    Yields:
      None.
    Raises:
       ValueError: if `value` is neither `0` nor `1`.
    """
    warnings.warn(
        "`tf.keras.backend.learning_phase_scope` is deprecated and "
        "will be removed after 2020-10-11. To update it, simply "
        "pass a True/False value to the `training` argument of the "
        "`__call__` method of your layer or model.",
        stacklevel=2,
    )
    with deprecated_internal_learning_phase_scope(value):
        try:
            yield
        finally:
            pass
