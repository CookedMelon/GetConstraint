@keras_export("keras.backend.set_learning_phase")
@doc_controls.do_not_generate_docs
def set_learning_phase(value):
    """Sets the learning phase to a fixed value.
    The backend learning phase affects any code that calls
    `backend.learning_phase()`
    In particular, all Keras built-in layers use the learning phase as the
    default for the `training` arg to `Layer.__call__`.
    User-written layers and models can achieve the same behavior with code that
    looks like:
    ```python
      def call(self, inputs, training=None):
        if training is None:
          training = backend.learning_phase()
    ```
    Args:
        value: Learning phase value, either 0 or 1 (integers).
               0 = test, 1 = train
    Raises:
        ValueError: if `value` is neither `0` nor `1`.
    """
    warnings.warn(
        "`tf.keras.backend.set_learning_phase` is deprecated and "
        "will be removed after 2020-10-11. To update it, simply "
        "pass a True/False value to the `training` argument of the "
        "`__call__` method of your layer or model."
    )
    deprecated_internal_set_learning_phase(value)
