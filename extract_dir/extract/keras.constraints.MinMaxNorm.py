@keras_export("keras.constraints.MinMaxNorm", "keras.constraints.min_max_norm")
class MinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.
    Also available via the shortcut function
    `tf.keras.constraints.min_max_norm`.
    Args:
      min_value: the minimum norm for the incoming weights.
      max_value: the maximum norm for the incoming weights.
      rate: rate for enforcing the constraint: weights will be
        rescaled to yield
        `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
        Effectively, this means that rate=1.0 stands for strict
        enforcement of the constraint, while rate<1.0 means that
        weights will be rescaled at each step to slowly move
        towards a value inside the desired interval.
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
    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis
    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        norms = backend.sqrt(
            tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True)
        )
        desired = (
            self.rate * backend.clip(norms, self.min_value, self.max_value)
            + (1 - self.rate) * norms
        )
        return w * (desired / (backend.epsilon() + norms))
    @doc_controls.do_not_generate_docs
    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate": self.rate,
            "axis": self.axis,
        }
