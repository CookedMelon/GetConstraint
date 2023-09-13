@keras_export("keras.estimator.model_to_estimator", v1=[])
def model_to_estimator_v2(
    keras_model=None,
    keras_model_path=None,
    custom_objects=None,
    model_dir=None,
    config=None,
    checkpoint_format="checkpoint",
    metric_names_map=None,
    export_outputs=None,
