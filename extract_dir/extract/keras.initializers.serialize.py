@keras_export("keras.initializers.serialize")
def serialize(initializer, use_legacy_format=False):
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(initializer)
    return serialization_lib.serialize_keras_object(initializer)
