@keras_export(
    "keras.applications.resnet_rs.ResNetRS270", "keras.applications.ResNetRS270"
)
def ResNetRS270(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS-270 model."""
    allow_bigger_recursion(1300)
    return ResNetRS(
        depth=270,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-270",
        include_preprocessing=include_preprocessing,
    )
