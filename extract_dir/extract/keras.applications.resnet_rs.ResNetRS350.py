@keras_export(
    "keras.applications.resnet_rs.ResNetRS350", "keras.applications.ResNetRS350"
)
def ResNetRS350(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Build ResNet-RS350 model."""
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.4,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-350",
        include_preprocessing=include_preprocessing,
    )
