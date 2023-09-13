"/home/cc/Workspace/tfconstraint/keras/saving/object_registration.py"
@keras_export(
    "keras.saving.register_keras_serializable",
    "keras.utils.register_keras_serializable",
)
def register_keras_serializable(package="Custom", name=None):
    """Registers an object with the Keras serialization framework.
    This decorator injects the decorated class or function into the Keras custom
    object dictionary, so that it can be serialized and deserialized without
    needing an entry in the user-provided custom object dict. It also injects a
    function that Keras will call to get the object's serializable string key.
    Note that to be serialized and deserialized, classes must implement the
    `get_config()` method. Functions do not have this requirement.
    The object will be registered under the key 'package>name' where `name`,
    defaults to the object name if not passed.
    Example:
    ```python
    # Note that `'my_package'` is used as the `package` argument here, and since
    # the `name` argument is not provided, `'MyDense'` is used as the `name`.
    @keras.saving.register_keras_serializable('my_package')
    class MyDense(keras.layers.Dense):
      pass
    assert keras.saving.get_registered_object('my_package>MyDense') == MyDense
    assert keras.saving.get_registered_name(MyDense) == 'my_package>MyDense'
    ```
    Args:
      package: The package that this class belongs to. This is used for the
        `key` (which is `"package>name"`) to idenfify the class. Note that this
        is the first argument passed into the decorator.
      name: The name to serialize this class under in this package. If not
        provided or `None`, the class' name will be used (note that this is the
        case when the decorator is used with only one argument, which becomes
        the `package`).
    Returns:
      A decorator that registers the decorated class with the passed names.
    """
    def decorator(arg):
        """Registers a class with the Keras serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + ">" + class_name
        if inspect.isclass(arg) and not hasattr(arg, "get_config"):
            raise ValueError(
                "Cannot register a class that does not have a "
                "get_config() method."
            )
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name
        return arg
    return decorator
