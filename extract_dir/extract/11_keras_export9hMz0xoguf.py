"/home/cc/Workspace/tfconstraint/keras/applications/resnet.py"
@keras_export(
    "keras.applications.resnet50.ResNet50",
    "keras.applications.resnet.ResNet50",
    "keras.applications.ResNet50",
)
def ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        return stack1(x, 512, 3, name="conv5")
    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
