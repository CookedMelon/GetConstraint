"/home/cc/Workspace/tfconstraint/keras/applications/efficientnet.py"
@keras_export(
    "keras.applications.efficientnet.EfficientNetB0",
    "keras.applications.EfficientNetB0",
)
def EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNet(
        1.0,
        1.0,
        224,
        0.2,
        model_name="efficientnetb0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
