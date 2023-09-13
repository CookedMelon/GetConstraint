@keras_export("keras.layers.Activation")
class Activation(Layer):
    """Applies an activation function to an output.
    Args:
      activation: Activation function, such as `tf.nn.relu`, or string name of
        built-in activation function, such as "relu".
    Usage:
    >>> layer = tf.keras.layers.Activation('relu')
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = tf.keras.layers.Activation(tf.nn.relu)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the batch axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as input.
    """
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)
    def call(self, inputs):
        return self.activation(inputs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {"activation": activations.serialize(self.activation)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
