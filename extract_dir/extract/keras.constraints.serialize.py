@keras_export("keras.constraints.serialize")
def serialize(constraint, use_legacy_format=False):
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(constraint)
    return serialize_keras_object(constraint)
