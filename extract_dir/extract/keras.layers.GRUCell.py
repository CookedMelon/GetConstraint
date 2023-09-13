@keras_export("keras.layers.GRUCell", v1=[])
class GRUCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    """Cell class for the GRU layer.
    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.
    This class processes one step within the whole time sequence input, whereas
    `tf.keras.layer.GRU` processes the whole sequence.
    For example:
    >>> inputs = tf.random.normal([32, 10, 8])
    >>> rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4))
    >>> output = rnn(inputs)
    >>> print(output.shape)
    (32, 4)
    >>> rnn = tf.keras.layers.RNN(
    ...    tf.keras.layers.GRUCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> print(whole_sequence_output.shape)
    (32, 10, 4)
    >>> print(final_state.shape)
    (32, 4)
    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state.  Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector. Default:
        `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and cuDNN compatible).
    Call arguments:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: A 2D tensor with shape of `[batch, units]`, which is the state
        from the previous time step. For timestep 0, the initial state provided
        by user will be feed to cell.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", True
            )
        else:
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", False
            )
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        implementation = kwargs.pop("implementation", 2)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        default_caching_device = rnn_utils.caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )
        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(2 * 3 * self.units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(
                shape=bias_shape,
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None
        self.built = True
    def call(self, inputs, states, training=None):
        h_tm1 = (
            states[0] if tf.nest.is_nested(states) else states
        )  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3
        )
        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)
        if self.implementation == 1:
            if 0.0 < self.dropout < 1.0:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
            x_z = backend.dot(inputs_z, self.kernel[:, : self.units])
            x_r = backend.dot(
                inputs_r, self.kernel[:, self.units : self.units * 2]
            )
            x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2 :])
            if self.use_bias:
                x_z = backend.bias_add(x_z, input_bias[: self.units])
                x_r = backend.bias_add(
                    x_r, input_bias[self.units : self.units * 2]
                )
                x_h = backend.bias_add(x_h, input_bias[self.units * 2 :])
            if 0.0 < self.recurrent_dropout < 1.0:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1
            recurrent_z = backend.dot(
                h_tm1_z, self.recurrent_kernel[:, : self.units]
            )
            recurrent_r = backend.dot(
                h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            if self.reset_after and self.use_bias:
                recurrent_z = backend.bias_add(
                    recurrent_z, recurrent_bias[: self.units]
                )
                recurrent_r = backend.bias_add(
                    recurrent_r, recurrent_bias[self.units : self.units * 2]
                )
            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = backend.dot(
                    h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )
                if self.use_bias:
                    recurrent_h = backend.bias_add(
                        recurrent_h, recurrent_bias[self.units * 2 :]
                    )
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = backend.dot(
                    r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )
            hh = self.activation(x_h + recurrent_h)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            # inputs projected by all gate matrices at once
            matrix_x = backend.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = backend.bias_add(matrix_x, input_bias)
            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)
            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = backend.bias_add(
                        matrix_inner, recurrent_bias
                    )
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = backend.dot(
                    h_tm1, self.recurrent_kernel[:, : 2 * self.units]
                )
            recurrent_z, recurrent_r, recurrent_h = tf.split(
                matrix_inner, [self.units, self.units, -1], axis=-1
            )
            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = backend.dot(
                    r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
                )
            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tf.nest.is_nested(states) else h
        return h, new_state
    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "implementation": self.implementation,
            "reset_after": self.reset_after,
        }
        config.update(rnn_utils.config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return rnn_utils.generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype
        )
