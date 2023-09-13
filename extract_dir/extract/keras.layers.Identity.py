@keras_export("keras.layers.Identity")
class Identity(Layer):
    """Identity layer.
    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.
    Args:
        name: Optional name for the layer instance.
    """
    def call(self, inputs):
        return tf.nest.map_structure(tf.identity, inputs)
