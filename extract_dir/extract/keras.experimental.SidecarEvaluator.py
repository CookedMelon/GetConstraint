@keras_export("keras.experimental.SidecarEvaluator", v1=[])
@deprecation.deprecated_endpoints("keras.experimental.SidecarEvaluator")
class SidecarEvaluatorExperimental(SidecarEvaluator):
    """Deprecated. Please use `tf.keras.utils.SidecarEvaluator` instead.
    Caution: `tf.keras.experimental.SidecarEvaluator` endpoint is
      deprecated and will be removed in a future release. Please use
      `tf.keras.utils.SidecarEvaluator`.
    """
    def __init__(self, *args, **kwargs):
        logging.warning(
            "`tf.keras.experimental.SidecarEvaluator` endpoint is "
            "deprecated and will be removed in a future release. Please use "
            "`tf.keras.utils.SidecarEvaluator`."
        )
        super().__init__(*args, **kwargs)
