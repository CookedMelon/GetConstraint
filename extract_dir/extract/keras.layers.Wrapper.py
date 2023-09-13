@keras_export("keras.layers.Wrapper")
class Wrapper(Layer):
    """Abstract wrapper base class.
    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
    Args:
      layer: The layer to be wrapped.
    """
    def __init__(self, layer, **kwargs):
        try:
            assert isinstance(layer, Layer)
        except Exception:
            raise ValueError(
                f"Layer {layer} supplied to wrapper is"
                " not a supported layer type. Please"
                " ensure wrapped layer is a valid Keras layer."
            )
        self.layer = layer
        super().__init__(**kwargs)
    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.built = True
    @property
    def activity_regularizer(self):
        if hasattr(self.layer, "activity_regularizer"):
            return self.layer.activity_regularizer
        else:
            return None
    def get_config(self):
        try:
            config = {
                "layer": serialization_lib.serialize_keras_object(self.layer)
            }
        except TypeError:  # Case of incompatible custom wrappers
            config = {
                "layer": legacy_serialization.serialize_keras_object(self.layer)
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer
        # Avoid mutating the input dict
        config = copy.deepcopy(config)
        use_legacy_format = "module" not in config
        layer = deserialize_layer(
            config.pop("layer"),
            custom_objects=custom_objects,
            use_legacy_format=use_legacy_format,
        )
        return cls(layer, **config)
