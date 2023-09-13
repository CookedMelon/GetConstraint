"/home/cc/Workspace/tfconstraint/keras/applications/regnet.py"
@keras_export(
    "keras.applications.regnet.RegNetX120", "keras.applications.RegNetX120"
)
def RegNetX120(
    model_name="regnetx120",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x120"]["depths"],
        MODEL_CONFIGS["x120"]["widths"],
        MODEL_CONFIGS["x120"]["group_width"],
        MODEL_CONFIGS["x120"]["block_type"],
        MODEL_CONFIGS["x120"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )
