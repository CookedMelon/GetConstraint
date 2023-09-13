@keras_export("keras.constraints.MaxNorm", "keras.constraints.max_norm")
class MaxNorm(Constraint):
    """MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.
    Also available via the shortcut function `tf.keras.constraints.max_norm`.
    Args:
      max_value: the maximum norm value for the incoming weights.
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
    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis
    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        norms = backend.sqrt(
            tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True)
        )
        desired = backend.clip(norms, 0, self.max_value)
        return w * (desired / (backend.epsilon() + norms))
    @doc_controls.do_not_generate_docs
    def get_config(self):
        return {"max_value": self.max_value, "axis": self.axis}
