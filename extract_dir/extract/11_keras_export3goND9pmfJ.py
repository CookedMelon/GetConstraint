"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/discretization.py"
@keras_export(
    "keras.layers.Discretization",
    "keras.layers.experimental.preprocessing.Discretization",
)
class Discretization(base_preprocessing_layer.PreprocessingLayer):
    """A preprocessing layer which buckets continuous features by ranges.
    This layer will place each element of its input data into one of several
    contiguous ranges and output an integer index indicating which range each
    element was placed in.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
      Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.
    Output shape:
      Same as input shape.
    Arguments:
      bin_boundaries: A list of bin boundaries. The leftmost and rightmost bins
        will always extend to `-inf` and `inf`, so `bin_boundaries=[0., 1., 2.]`
        generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
        If this option is set, `adapt()` should not be called.
      num_bins: The integer number of bins to compute. If this option is set,
        `adapt()` should be called to learn the bin boundaries.
      epsilon: Error tolerance, typically a small fraction close to zero (e.g.
        0.01). Higher values of epsilon increase the quantile approximation, and
        hence result in more unequal buckets, but could improve performance
        and resource consumption.
      output_mode: Specification for the output of the layer. Values can be
       `"int"`, `"one_hot"`, `"multi_hot"`, or
        `"count"` configuring the layer as follows:
          - `"int"`: Return the discretized bin indices directly.
          - `"one_hot"`: Encodes each individual element in the input into an
            array the same size as `num_bins`, containing a 1 at the input's bin
            index. If the last dimension is size 1, will encode on that
            dimension.  If the last dimension is not size 1, will append a new
            dimension for the encoded output.
          - `"multi_hot"`: Encodes each sample in the input into a single array
            the same size as `num_bins`, containing a 1 for each bin index
            index present in the sample. Treats the last dimension as the sample
            dimension, if input shape is `(..., sample_length)`, output shape
            will be `(..., num_tokens)`.
          - `"count"`: As `"multi_hot"`, but the int array contains a count of
            the number of times the bin index appeared in the sample.
        Defaults to `"int"`.
      sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
        and `"count"` output modes. If True, returns a `SparseTensor` instead of
        a dense `Tensor`. Defaults to `False`.
    Examples:
    Bucketize float values based on provided buckets.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = tf.keras.layers.Discretization(bin_boundaries=[0., 1., 2.])
    >>> layer(input)
    <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
    array([[0, 2, 3, 1],
           [1, 3, 2, 1]])>
    Bucketize float values based on a number of buckets to compute.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = tf.keras.layers.Discretization(num_bins=4, epsilon=0.01)
    >>> layer.adapt(input)
    >>> layer(input)
    <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
    array([[0, 2, 3, 2],
           [1, 3, 3, 1]])>
    """
    def __init__(
        self,
        bin_boundaries=None,
        num_bins=None,
        epsilon=0.01,
        output_mode="int",
        sparse=False,
        **kwargs,
    ):
        # bins is a deprecated arg for setting bin_boundaries or num_bins that
        # still has some usage.
        if "bins" in kwargs:
            logging.warning(
                "bins is deprecated, "
                "please use bin_boundaries or num_bins instead."
            )
            if isinstance(kwargs["bins"], int) and num_bins is None:
                num_bins = kwargs["bins"]
            elif bin_boundaries is None:
                bin_boundaries = kwargs["bins"]
            del kwargs["bins"]
        # By default, output int64 when output_mode='int' and floats otherwise.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = (
                tf.int64 if output_mode == INT else backend.floatx()
            )
        elif (
            output_mode == "int" and not tf.as_dtype(kwargs["dtype"]).is_integer
        ):
            # Compat for when dtype was always floating and ignored by the
            # layer.
            kwargs["dtype"] = tf.int64
        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("Discretization").set(
            True
        )
        # Check dtype only after base layer parses it; dtype parsing is complex.
        if (
            output_mode == INT
            and not tf.as_dtype(self.compute_dtype).is_integer
        ):
            input_dtype = kwargs["dtype"]
            raise ValueError(
                "When `output_mode='int'`, `dtype` should be an integer "
                f"type. Received: dtype={input_dtype}"
            )
        # 'output_mode' must be one of (INT, ONE_HOT, MULTI_HOT, COUNT)
        layer_utils.validate_string_arg(
            output_mode,
            allowable_strings=(INT, ONE_HOT, MULTI_HOT, COUNT),
            layer_name=self.__class__.__name__,
            arg_name="output_mode",
        )
        if sparse and output_mode == INT:
            raise ValueError(
                "`sparse` may only be true if `output_mode` is "
                "`'one_hot'`, `'multi_hot'`, or `'count'`. "
                f"Received: sparse={sparse} and "
                f"output_mode={output_mode}"
            )
        if num_bins is not None and num_bins < 0:
            raise ValueError(
                "`num_bins` must be greater than or equal to 0. "
                "You passed `num_bins={}`".format(num_bins)
            )
        if num_bins is not None and bin_boundaries is not None:
            raise ValueError(
                "Both `num_bins` and `bin_boundaries` should not be "
                "set. You passed `num_bins={}` and "
                "`bin_boundaries={}`".format(num_bins, bin_boundaries)
            )
        bin_boundaries = utils.listify_tensors(bin_boundaries)
        self.input_bin_boundaries = bin_boundaries
        self.bin_boundaries = (
            bin_boundaries if bin_boundaries is not None else []
        )
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.output_mode = output_mode
        self.sparse = sparse
    def build(self, input_shape):
        super().build(input_shape)
        if self.input_bin_boundaries is not None:
            return
        # Summary contains two equal length vectors of bins at index 0 and
        # weights at index 1.
        self.summary = self.add_weight(
            name="summary",
            shape=(2, None),
            dtype=tf.float32,
            initializer=lambda shape, dtype: [
                [],
                [],
            ],
            trainable=False,
        )
    # We override this method solely to generate a docstring.
    def adapt(self, data, batch_size=None, steps=None):
        """Computes bin boundaries from quantiles in a input dataset.
        Calling `adapt()` on a `Discretization` layer is an alternative to
        passing in a `bin_boundaries` argument during construction. A
        `Discretization` layer should always be either adapted over a dataset or
        passed `bin_boundaries`.
        During `adapt()`, the layer will estimate the quantile boundaries of the
        input dataset. The number of quantiles can be controlled via the
        `num_bins` argument, and the error tolerance for quantile boundaries can
        be controlled via the `epsilon` argument.
        In order to make `Discretization` efficient in any distribution context,
        the computed boundaries are kept static with respect to any compiled
        `tf.Graph`s that call the layer. As a consequence, if the layer is
        adapted a second time, any models using the layer should be re-compiled.
        For more information see
        `tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt`.
        `adapt()` is meant only as a single machine utility to compute layer
        state.  To analyze a dataset that cannot fit on a single machine, see
        [Tensorflow Transform](
        https://www.tensorflow.org/tfx/transform/get_started) for a
        multi-machine, map-reduce solution.
        Arguments:
          data: The data to train on. It can be passed either as a
              `tf.data.Dataset`, or as a numpy array.
          batch_size: Integer or `None`.
              Number of samples per state update.
              If unspecified, `batch_size` will default to 32.
              Do not specify the `batch_size` if your data is in the
              form of datasets, generators, or `keras.utils.Sequence` instances
              (since they generate batches).
          steps: Integer or `None`.
              Total number of steps (batches of samples)
              When training with input tensors such as
              TensorFlow data tensors, the default `None` is equal to
              the number of samples in your dataset divided by
              the batch size, or 1 if that cannot be determined. If x is a
              `tf.data` dataset, and 'steps' is None, the epoch will run until
              the input dataset is exhausted. When passing an infinitely
              repeating dataset, you must specify the `steps` argument. This
              argument is not supported with array inputs.
        """
        super().adapt(data, batch_size=batch_size, steps=steps)
    def update_state(self, data):
        if self.input_bin_boundaries is not None:
            raise ValueError(
                "Cannot adapt a Discretization layer that has been initialized "
                "with `bin_boundaries`, use `num_bins` instead. You passed "
                "`bin_boundaries={}`.".format(self.input_bin_boundaries)
            )
        if not self.built:
            raise RuntimeError("`build` must be called before `update_state`.")
        data = tf.convert_to_tensor(data)
        if data.dtype != tf.float32:
            data = tf.cast(data, tf.float32)
        summary = summarize(data, self.epsilon)
        self.summary.assign(
            merge_summaries(summary, self.summary, self.epsilon)
        )
    def finalize_state(self):
        if self.input_bin_boundaries is not None or not self.built:
            return
        # The bucketize op only support list boundaries.
        self.bin_boundaries = utils.listify_tensors(
            get_bin_boundaries(self.summary, self.num_bins)
        )
    def reset_state(self):
        if self.input_bin_boundaries is not None or not self.built:
            return
        self.summary.assign([[], []])
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bin_boundaries": self.input_bin_boundaries,
                "num_bins": self.num_bins,
                "epsilon": self.epsilon,
                "output_mode": self.output_mode,
                "sparse": self.sparse,
            }
        )
        return config
    def compute_output_shape(self, input_shape):
        return input_shape
    def compute_output_signature(self, input_spec):
        output_shape = self.compute_output_shape(input_spec.shape.as_list())
        if isinstance(input_spec, tf.SparseTensorSpec):
            return tf.SparseTensorSpec(
                shape=output_shape, dtype=self.compute_dtype
            )
        return tf.TensorSpec(shape=output_shape, dtype=self.compute_dtype)
    def call(self, inputs):
        def bucketize(inputs):
            return tf.raw_ops.Bucketize(
                input=inputs, boundaries=self.bin_boundaries
            )
        if tf_utils.is_ragged(inputs):
            indices = tf.ragged.map_flat_values(bucketize, inputs)
        elif tf_utils.is_sparse(inputs):
            indices = tf.SparseTensor(
                indices=tf.identity(inputs.indices),
                values=bucketize(inputs.values),
                dense_shape=tf.identity(inputs.dense_shape),
            )
        else:
            indices = bucketize(inputs)
        return utils.encode_categorical_inputs(
            indices,
            output_mode=self.output_mode,
            depth=len(self.bin_boundaries) + 1,
            sparse=self.sparse,
            dtype=self.compute_dtype,
        )
