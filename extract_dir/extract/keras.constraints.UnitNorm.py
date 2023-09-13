@keras_export("keras.constraints.UnitNorm", "keras.constraints.unit_norm")
class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    Also available via the shortcut function `tf.keras.constraints.unit_norm`.
    Args:
      axis: integer, axis along which to calculate weight norms.
        For instance, in a `Dense` layer the weight matrix
        has shape `(input_dim, output_dim)`,
        set `axis` to `0` to constrain each weight vector
        of length `(input_dim,)`.
        In a `Conv2D` layer with `data_format="channels_last"`,
        the weight tensor has shape
        `(rows, cols, input_depth, output_depth)`,
        set `axis` to `[0, 1, 2]`
        to constrain the weights of each filter tensor of size
        `(rows, cols, input_depth)`.
    """
    def __init__(self, axis=0):
        self.axis = axis
    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        return w / (
            backend.epsilon()
            + backend.sqrt(
                tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True)
            )
        )
    @doc_controls.do_not_generate_docs
    def get_config(self):
        return {"axis": self.axis}
