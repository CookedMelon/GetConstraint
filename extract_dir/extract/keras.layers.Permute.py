@keras_export("keras.layers.Permute")
class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.
    Useful e.g. connecting RNNs and convnets.
    Example:
    ```python
    model = Sequential()
    model.add(Permute((2, 1), input_shape=(10, 64)))
    # now: model.output_shape == (None, 64, 10)
    # note: `None` is the batch dimension
    ```
    Args:
      dims: Tuple of integers. Permutation pattern does not include the
        samples dimension. Indexing starts at 1.
        For instance, `(2, 1)` permutes the first and second dimensions
        of the input.
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same as the input shape, but with the dimensions re-ordered according
      to the specified pattern.
    """
    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)
        self.dims = tuple(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError(
                "Invalid permutation argument `dims` for Permute Layer. "
                "The set of indices in `dims` must be consecutive and start "
                f"from 1. Received dims={dims}"
            )
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i + 1] = target_dim
        return tf.TensorShape(output_shape)
    def call(self, inputs):
        return tf.transpose(inputs, perm=(0,) + self.dims)
    def get_config(self):
        config = {"dims": self.dims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
