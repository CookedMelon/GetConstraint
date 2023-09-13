"/home/cc/Workspace/tfconstraint/keras/applications/resnet.py"
@keras_export(
    "keras.applications.resnet.ResNet152", "keras.applications.ResNet152"
)
def ResNet152(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs,
):
    """Instantiates the ResNet152 architecture."""
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 8, name="conv3")
        x = stack1(x, 256, 36, name="conv4")
        return stack1(x, 512, 3, name="conv5")
    return ResNet(
        stack_fn,
        False,
        True,
        "resnet152",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
