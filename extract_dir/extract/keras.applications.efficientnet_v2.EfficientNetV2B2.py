@keras_export(
    "keras.applications.efficientnet_v2.EfficientNetV2B2",
    "keras.applications.EfficientNetV2B2",
)
def EfficientNetV2B2(
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
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        model_name="efficientnetv2-b2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )
