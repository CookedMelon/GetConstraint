@keras_export(
    "keras.applications.resnet_rs.ResNetRS101", "keras.applications.ResNetRS101"
)
def ResNetRS101(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS101 model."""
    return ResNetRS(
        depth=101,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-101",
        include_preprocessing=include_preprocessing,
    )
