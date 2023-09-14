@keras_export(
    "keras.saving.custom_object_scope",
    "keras.utils.custom_object_scope",
    "keras.utils.CustomObjectScope",
)
class CustomObjectScope:
    """Exposes custom classes/functions to Keras deserialization internals.
    Under a scope `with custom_object_scope(objects_dict)`, Keras methods such
    as `tf.keras.models.load_model` or `tf.keras.models.model_from_config`
    will be able to deserialize any custom object referenced by a
    saved config (e.g. a custom layer or metric).
    Example:
    Consider a custom regularizer `my_regularizer`:
    ```python
    layer = Dense(3, kernel_regularizer=my_regularizer)
    # Config contains a reference to `my_regularizer`
    config = layer.get_config()
    ...
    # Later:
    with custom_object_scope({'my_regularizer': my_regularizer}):
      layer = Dense.from_config(config)
    ```
    Args:
        *args: Dictionary or dictionaries of `{name: object}` pairs.
    """
    def __init__(self, *args):
        self.custom_objects = args
        self.backup = None
    def __enter__(self):
        self.backup = _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.copy()
        for objects in self.custom_objects:
            _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.update(objects)
        return self
    def __exit__(self, *args, **kwargs):
        _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.clear()
        _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.update(self.backup)
