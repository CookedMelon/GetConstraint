"/home/cc/Workspace/tfconstraint/keras/saving/object_registration.py"
@keras_export(
    "keras.saving.get_registered_name", "keras.utils.get_registered_name"
)
def get_registered_name(obj):
    """Returns the name registered to an object within the Keras framework.
    This function is part of the Keras serialization and deserialization
    framework. It maps objects to the string names associated with those objects
    for serialization/deserialization.
    Args:
      obj: The object to look up.
    Returns:
      The name associated with the object, or the default Python name if the
        object is not registered.
    """
    if obj in _GLOBAL_CUSTOM_NAMES:
        return _GLOBAL_CUSTOM_NAMES[obj]
    else:
        return obj.__name__
