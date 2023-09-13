@keras_export("keras.activations.deserialize")
@tf.__internal__.dispatch.add_dispatch_support
def deserialize(name, custom_objects=None, use_legacy_format=False):
    """Returns activation function given a string identifier.
    Args:
      name: The name of the activation function.
      custom_objects: Optional `{function_name: function_obj}`
        dictionary listing user-provided activation functions.
    Returns:
        Corresponding activation function.
    Example:
    >>> tf.keras.activations.deserialize('linear')
     <function linear at 0x1239596a8>
    >>> tf.keras.activations.deserialize('sigmoid')
     <function sigmoid at 0x123959510>
    >>> tf.keras.activations.deserialize('abcd')
    Traceback (most recent call last):
    ...
    ValueError: Unknown activation function 'abcd' cannot be deserialized.
    Raises:
        ValueError: `Unknown activation function` if the input string does not
        denote any defined Tensorflow activation function.
    """
    activation_functions = {}
    current_module = sys.modules[__name__]
    # we put 'current_module' after 'activation_layers' to prefer the local one
    # if there is a collision
    generic_utils.populate_dict_with_module_objects(
        activation_functions,
        (activation_layers, current_module),
        obj_filter=callable,
    )
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            name,
            module_objects=activation_functions,
            custom_objects=custom_objects,
            printable_module_name="activation function",
        )
    returned_fn = serialization_lib.deserialize_keras_object(
        name,
        module_objects=activation_functions,
        custom_objects=custom_objects,
        printable_module_name="activation function",
    )
    if isinstance(returned_fn, str):
        raise ValueError(
            f"Unknown activation function '{name}' cannot be deserialized."
        )
    return returned_fn
