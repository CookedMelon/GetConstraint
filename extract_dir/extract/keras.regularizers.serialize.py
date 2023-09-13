@keras_export("keras.regularizers.serialize")
def serialize(regularizer, use_legacy_format=False):
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(regularizer)
    return serialize_keras_object(regularizer)
