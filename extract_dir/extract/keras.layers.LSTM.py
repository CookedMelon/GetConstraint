@keras_export("keras.layers.LSTM", v1=[])
class LSTM(DropoutRNNCellMixin, RNN, base_layer.BaseRandomLayer):
    """Long Short-Term Memory layer - Hochreiter 1997.
    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.
    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or pure-TensorFlow)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation.
    The requirements to use the cuDNN implementation are:
    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. Inputs, if use masking, are strictly right-padded.
    7. Eager execution is enabled in the outermost context.
    For example:
    >>> inputs = tf.random.normal([32, 10, 8])
    >>> lstm = tf.keras.layers.LSTM(4)
    >>> output = lstm(inputs)
    >>> print(output.shape)
    (32, 4)
    >>> lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
    >>> print(whole_seq_output.shape)
    (32, 10, 4)
    >>> print(final_memory_state.shape)
    (32, 4)
    >>> print(final_carry_state.shape)
    (32, 4)
    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
        is applied (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs. Default: `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
        Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
        the forget gate at initialization. Setting it to true will also force
        `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
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
      return_sequences: Boolean. Whether to return the last output in the output
        sequence, or the full sequence. Default: `False`.
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `False`.
      go_backwards: Boolean (default `False`). If True, process the input
        sequence backwards and return the reversed sequence.
      stateful: Boolean (default `False`). If True, the last state for each
      sample at index i in a batch will be used as initial state for the sample
        of index i in the following batch.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `[timesteps, batch, feature]`, whereas in the False case, it will be
        `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.
      unroll: Boolean (default `False`). If True, the network will be unrolled,
        else a symbolic loop will be used. Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive. Unrolling is only
        suitable for short sequences.
    Call arguments:
      inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
      mask: Binary tensor of shape `[batch, timesteps]` indicating whether
        a given timestep should be masked (optional).
        An individual `True` entry indicates that the corresponding timestep
        should be utilized, while a `False` entry indicates that the
        corresponding timestep should be ignored. Defaults to `None`.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used (optional). Defaults to `None`.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell (optional, `None` causes creation
        of zero-filled initial state tensors). Defaults to `None`.
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
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False,
        unroll=False,
        **kwargs,
    ):
        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self.return_runtime = kwargs.pop("return_runtime", False)
        implementation = kwargs.pop("implementation", 2)
        if implementation == 0:
            logging.warning(
                "`implementation=0` has been deprecated, "
                "and now defaults to `implementation=1`."
                "Please update your layer call."
            )
        if "enable_caching_device" in kwargs:
            cell_kwargs = {
                "enable_caching_device": kwargs.pop("enable_caching_device")
            }
        else:
            cell_kwargs = {}
        cell = LSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
            name="lstm_cell",
            **cell_kwargs,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            time_major=time_major,
            unroll=unroll,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = [
            InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
        ]
        self._could_use_gpu_kernel = (
            self.activation in (activations.tanh, tf.tanh)
            and self.recurrent_activation in (activations.sigmoid, tf.sigmoid)
            and recurrent_dropout == 0
            and not unroll
            and use_bias
            and tf.compat.v1.executing_eagerly_outside_functions()
        )
        if tf.config.list_logical_devices("GPU"):
            # Only show the message when there is GPU available, user will not
            # care about the cuDNN if there isn't any GPU.
            if self._could_use_gpu_kernel:
                logging.debug(gru_lstm_utils.CUDNN_AVAILABLE_MSG % self.name)
            else:
                logging.warning(
                    gru_lstm_utils.CUDNN_NOT_AVAILABLE_MSG % self.name
                )
        if gru_lstm_utils.use_new_gru_lstm_impl():
            self._defun_wrapper = gru_lstm_utils.DefunWrapper(
                time_major, go_backwards, "lstm"
            )
    def call(self, inputs, mask=None, training=None, initial_state=None):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
        is_ragged_input = row_lengths is not None
        self._validate_args_if_ragged(is_ragged_input, mask)
        # LSTM does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(
            inputs, initial_state, None
        )
        if isinstance(mask, list):
            mask = mask[0]
        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        if not self._could_use_gpu_kernel:
            # Fall back to use the normal LSTM.
            kwargs = {"training": training}
            self._maybe_reset_cell_dropout_mask(self.cell)
            def step(inputs, states):
                return self.cell(inputs, states, **kwargs)
            last_output, outputs, states = backend.rnn(
                step,
                inputs,
                initial_state,
                constants=None,
                go_backwards=self.go_backwards,
                mask=mask,
                unroll=self.unroll,
                input_length=row_lengths
                if row_lengths is not None
                else timesteps,
                time_major=self.time_major,
                zero_output_for_mask=self.zero_output_for_mask,
                return_all_outputs=self.return_sequences,
            )
            runtime = gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_UNKNOWN)
        else:
            # Use the new defun approach for backend implementation swap.
            # Note that different implementations need to have same function
            # signature, eg, the tensor parameters need to have same shape and
            # dtypes. Since the cuDNN has an extra set of bias, those bias will
            # be passed to both normal and cuDNN implementations.
            self.reset_dropout_mask()
            dropout_mask = self.get_dropout_mask_for_cell(
                inputs, training, count=4
            )
            if dropout_mask is not None:
                inputs = inputs * dropout_mask[0]
            if gru_lstm_utils.use_new_gru_lstm_impl():
                lstm_kwargs = {
                    "inputs": inputs,
                    "init_h": gru_lstm_utils.read_variable_value(
                        initial_state[0]
                    ),
                    "init_c": gru_lstm_utils.read_variable_value(
                        initial_state[1]
                    ),
                    "kernel": gru_lstm_utils.read_variable_value(
                        self.cell.kernel
                    ),
                    "recurrent_kernel": gru_lstm_utils.read_variable_value(
                        self.cell.recurrent_kernel
                    ),
                    "bias": gru_lstm_utils.read_variable_value(self.cell.bias),
                    "mask": mask,
                    "time_major": self.time_major,
                    "go_backwards": self.go_backwards,
                    "sequence_lengths": row_lengths,
                    "zero_output_for_mask": self.zero_output_for_mask,
                }
                (
                    last_output,
                    outputs,
                    new_h,
                    new_c,
                    runtime,
                ) = self._defun_wrapper.defun_layer(**lstm_kwargs)
            else:
                gpu_lstm_kwargs = {
                    "inputs": inputs,
                    "init_h": gru_lstm_utils.read_variable_value(
                        initial_state[0]
                    ),
                    "init_c": gru_lstm_utils.read_variable_value(
                        initial_state[1]
                    ),
                    "kernel": gru_lstm_utils.read_variable_value(
                        self.cell.kernel
                    ),
                    "recurrent_kernel": gru_lstm_utils.read_variable_value(
                        self.cell.recurrent_kernel
                    ),
                    "bias": gru_lstm_utils.read_variable_value(self.cell.bias),
                    "mask": mask,
                    "time_major": self.time_major,
                    "go_backwards": self.go_backwards,
                    "sequence_lengths": row_lengths,
                    "return_sequences": self.return_sequences,
                }
                normal_lstm_kwargs = gpu_lstm_kwargs.copy()
                normal_lstm_kwargs.update(
                    {
                        "zero_output_for_mask": self.zero_output_for_mask,
                    }
                )
                if tf.executing_eagerly():
                    device_type = gru_lstm_utils.get_context_device_type()
                    can_use_gpu = (
                        # Either user specified GPU or unspecified but GPU is
                        # available.
                        (
                            device_type == gru_lstm_utils.GPU_DEVICE_NAME
                            or (
                                device_type is None
                                and tf.config.list_logical_devices("GPU")
                            )
                        )
                        and gru_lstm_utils.is_cudnn_supported_inputs(
                            mask, self.time_major, row_lengths
                        )
                    )
                    # Under eager context, check the device placement and prefer
                    # the GPU implementation when GPU is available.
                    if can_use_gpu:
                        last_output, outputs, new_h, new_c, runtime = gpu_lstm(
                            **gpu_lstm_kwargs
                        )
                    else:
                        (
                            last_output,
                            outputs,
                            new_h,
                            new_c,
                            runtime,
                        ) = standard_lstm(**normal_lstm_kwargs)
                else:
                    (
                        last_output,
                        outputs,
                        new_h,
                        new_c,
                        runtime,
                    ) = lstm_with_backend_selection(**normal_lstm_kwargs)
            states = [new_h, new_c]
        if self.stateful:
            updates = [
                tf.compat.v1.assign(
                    self_state, tf.cast(state, self_state.dtype)
                )
                for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)
        if self.return_sequences:
            output = backend.maybe_convert_to_ragged(
                is_ragged_input,
                outputs,
                row_lengths,
                go_backwards=self.go_backwards,
            )
        else:
            output = last_output
        if self.return_state:
            return [output] + list(states)
        elif self.return_runtime:
            return output, runtime
        else:
            return output
    @property
    def units(self):
        return self.cell.units
    @property
    def activation(self):
        return self.cell.activation
    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation
    @property
    def use_bias(self):
        return self.cell.use_bias
    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer
    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer
    @property
    def bias_initializer(self):
        return self.cell.bias_initializer
    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias
    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer
    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer
    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer
    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint
    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint
    @property
    def bias_constraint(self):
        return self.cell.bias_constraint
    @property
    def dropout(self):
        return self.cell.dropout
    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout
    @property
    def implementation(self):
        return self.cell.implementation
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
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "implementation": self.implementation,
        }
        config.update(rnn_utils.config_for_enable_caching_device(self.cell))
        base_config = super().get_config()
        del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))
    @classmethod
    def from_config(cls, config):
        if "implementation" in config and config["implementation"] == 0:
            config["implementation"] = 1
        return cls(**config)
