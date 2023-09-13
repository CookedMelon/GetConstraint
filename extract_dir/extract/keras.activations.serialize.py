@keras_export("keras.activations.serialize")
@tf.__internal__.dispatch.add_dispatch_support
def serialize(activation, use_legacy_format=False):
    """Returns the string identifier of an activation function.
    Args:
        activation : Function object.
    Returns:
        String denoting the name attribute of the input function
    Example:
    >>> tf.keras.activations.serialize(tf.keras.activations.tanh)
    'tanh'
    >>> tf.keras.activations.serialize(tf.keras.activations.sigmoid)
    'sigmoid'
    >>> tf.keras.activations.serialize('abcd')
    Traceback (most recent call last):
    ...
    ValueError: Unknown activation function 'abcd' cannot be serialized.
    Raises:
        ValueError: The input function is not a valid one.
    """
    if (
        hasattr(activation, "__name__")
        and activation.__name__ in _TF_ACTIVATIONS_V2
    ):
        return _TF_ACTIVATIONS_V2[activation.__name__]
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(activation)
    fn_config = serialization_lib.serialize_keras_object(activation)
    if (
        not tf.__internal__.tf2.enabled()
        or saved_model_utils.in_tf_saved_model_scope()
    ):
        return fn_config
    if "config" not in fn_config:
        raise ValueError(
            f"Unknown activation function '{activation}' cannot be "
            "serialized due to invalid function name. Make sure to use "
            "an activation name that matches the references defined in "
            "activations.py or use "
            "`@keras.saving.register_keras_serializable()` "
            "to register any custom activations. "
            f"config={fn_config}"
        )
    if not isinstance(activation, types.FunctionType):
        # Case for additional custom activations represented by objects
        return fn_config
    if (
        isinstance(fn_config["config"], str)
        and fn_config["config"] not in globals()
    ):
        # Case for custom activation functions from external activations modules
        fn_config["config"] = object_registration.get_registered_name(
            activation
        )
        return fn_config
    return fn_config["config"]
    # Case for keras.activations builtins (simply return name)
