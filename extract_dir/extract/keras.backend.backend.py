@keras_export("keras.backend.backend")
@doc_controls.do_not_generate_docs
def backend():
    """Publicly accessible method for determining the current backend.
    Only exists for API compatibility with multi-backend Keras.
    Returns:
        The string "tensorflow".
    """
    return "tensorflow"
