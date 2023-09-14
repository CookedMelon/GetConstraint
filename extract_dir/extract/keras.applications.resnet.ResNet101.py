@keras_export(
    "keras.applications.resnet.ResNet101", "keras.applications.ResNet101"
)
def ResNet101(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs,
):
    """Instantiates the ResNet101 architecture."""
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 23, name="conv4")
        return stack1(x, 512, 3, name="conv5")
    return ResNet(
        stack_fn,
        False,
        True,
        "resnet101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
