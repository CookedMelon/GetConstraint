"/home/cc/Workspace/tfconstraint/keras/applications/regnet.py"
@keras_export(
    "keras.applications.regnet.RegNetY016", "keras.applications.RegNetY016"
)
def RegNetY016(
    model_name="regnety016",
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
        MODEL_CONFIGS["y016"]["depths"],
        MODEL_CONFIGS["y016"]["widths"],
        MODEL_CONFIGS["y016"]["group_width"],
        MODEL_CONFIGS["y016"]["block_type"],
        MODEL_CONFIGS["y016"]["default_size"],
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
