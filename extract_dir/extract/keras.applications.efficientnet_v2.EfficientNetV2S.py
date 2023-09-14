@keras_export(
    "keras.applications.efficientnet_v2.EfficientNetV2S",
    "keras.applications.EfficientNetV2S",
)
def EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=384,
        model_name="efficientnetv2-s",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )
