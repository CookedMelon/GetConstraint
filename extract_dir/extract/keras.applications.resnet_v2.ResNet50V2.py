@keras_export(
    "keras.applications.resnet_v2.ResNet50V2", "keras.applications.ResNet50V2"
)
def ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the ResNet50V2 architecture."""
    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 4, name="conv3")
        x = resnet.stack2(x, 256, 6, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")
    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet50v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )
