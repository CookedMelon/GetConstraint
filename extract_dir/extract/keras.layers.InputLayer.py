@keras_export("keras.layers.InputLayer")
class InputLayer(base_layer.Layer):
    """Layer to be used as an entry point into a Network (a graph of layers).
    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create a placeholder tensor (pass arguments `input_shape`, and
    optionally, `dtype`).
    It is generally recommend to use the Keras Functional model via `Input`,
    (which creates an `InputLayer`) without directly using `InputLayer`.
    When using `InputLayer` with the Keras Sequential model, it can be skipped
    by moving the `input_shape` parameter to the first layer after the
    `InputLayer`.
    This class can create placeholders for `tf.Tensors`, `tf.SparseTensors`, and
    `tf.RaggedTensors` by choosing `sparse=True` or `ragged=True`. Note that
    `sparse` and `ragged` can't be configured to `True` at the same time.
    Usage:
    ```python
    # With explicit InputLayer.
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(4,)),
      tf.keras.layers.Dense(8)])
    model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
    model.fit(np.zeros((10, 4)),
              np.ones((10, 8)))
    # Without InputLayer and let the first layer to have the input_shape.
    # Keras will add a input for the model behind the scene.
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(8, input_shape=(4,))])
    model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
    model.fit(np.zeros((10, 4)),
              np.ones((10, 8)))
    ```
    Args:
        input_shape: Shape tuple (not including the batch axis), or
            `TensorShape` instance (not including the batch axis).
        batch_size: Optional input batch size (integer or `None`).
        dtype: Optional datatype of the input. When not provided, the Keras
            default `float` type will be used.
        input_tensor: Optional tensor to use as layer input. If set, the layer
            will use the `tf.TypeSpec` of this tensor rather
            than creating a new placeholder tensor.
        sparse: Boolean, whether the placeholder created is meant to be sparse.
            Defaults to `False`.
        ragged: Boolean, whether the placeholder created is meant to be ragged.
            In this case, values of `None` in the `shape` argument represent
            ragged dimensions. For more information about `tf.RaggedTensor`, see
            [this guide](https://www.tensorflow.org/guide/ragged_tensor).
            Defaults to `False`.
        type_spec: A `tf.TypeSpec` object to create Input from. This
            `tf.TypeSpec` represents the entire batch. When provided, all other
            args except name must be `None`.
        name: Optional name of the layer (string).
    """
    @traceback_utils.filter_traceback
    def __init__(
        self,
        input_shape=None,
        batch_size=None,
        dtype=None,
        input_tensor=None,
        sparse=None,
        name=None,
        ragged=None,
        type_spec=None,
        **kwargs,
    ):
        self._init_input_shape = input_shape
        self._init_batch_size = batch_size
        self._init_dtype = dtype
        self._init_sparse = sparse
        self._init_ragged = ragged
        self._init_type_spec = type_spec
        strategy = tf.distribute.get_strategy()
        if (
            strategy
            and batch_size is not None
            and distributed_training_utils.global_batch_size_supported(strategy)
        ):
            if batch_size % strategy.num_replicas_in_sync != 0:
                raise ValueError(
                    "The `batch_size` argument ({}) must be divisible by "
                    "the number of replicas ({})".format(
                        batch_size, strategy.num_replicas_in_sync
                    )
                )
            batch_size = batch_size // strategy.num_replicas_in_sync
        if "batch_input_shape" in kwargs:
            batch_input_shape = kwargs.pop("batch_input_shape")
            if input_shape and batch_input_shape:
                raise ValueError(
                    "Only provide the input_shape OR "
                    "batch_input_shape argument to "
                    "InputLayer, not both at the same time."
                )
            # Set the input shape and batch size from the batch_input_shape.
            # Note that batch_input_shape can be None (unknown rank) or []
            # (scalar), in which case the batch size must be None.
            if batch_input_shape:
                batch_size = batch_input_shape[0]
                input_shape = batch_input_shape[1:]
        if kwargs:
            raise ValueError(
                f"Unrecognized keyword arguments: {list(kwargs.keys())}"
            )
        if sparse and ragged:
            raise ValueError(
                "Cannot set both sparse and ragged to True in a Keras input."
            )
        if not name:
            prefix = "input"
            name = prefix + "_" + str(backend.get_uid(prefix))
        if not dtype:
            if input_tensor is None:
                dtype = backend.floatx()
            else:
                dtype = backend.dtype(input_tensor)
        elif input_tensor is not None and input_tensor.dtype != dtype:
            raise ValueError(
                "`input_tensor.dtype` differs from `dtype`. Received: "
                f"input_tensor.dtype={input_tensor.dtype} "
                f"but expected dtype={dtype}"
            )
        super().__init__(dtype=dtype, name=name)
        self.built = True
        self.sparse = True if sparse else False
        self.ragged = True if ragged else False
        self.batch_size = batch_size
        self.supports_masking = True
        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(input_shape.as_list())
        elif isinstance(input_shape, int):
            input_shape = (input_shape,)
        if type_spec is not None:
            args_that_must_be_none = [
                ("(input_)shape", self._init_input_shape),
                ("batch_size", self._init_batch_size),
                ("dtype", self._init_dtype),
                ("input_tensor", input_tensor),
                ("sparse", self._init_sparse),
                ("ragged", self._init_ragged),
            ]
            for arg_name, arg in args_that_must_be_none:
                _assert_other_arg_none(arg_name, arg)
            if not tf.compat.v1.executing_eagerly_outside_functions():
                raise ValueError(
                    "Creating Keras inputs from a type_spec is only "
                    "supported when eager execution is enabled."
                )
            # Needed for type_spec deserialization since TypeSpec objects
            # are not Keras-native (not automatically deserialized).
            if isinstance(type_spec, dict):
                type_spec = serialization_lib.deserialize_keras_object(
                    type_spec
                )
            input_tensor = keras_tensor.keras_tensor_from_type_spec(type_spec)
            if isinstance(input_tensor, keras_tensor.SparseKerasTensor):
                self.sparse = True
            if isinstance(input_tensor, keras_tensor.RaggedKerasTensor):
                self.ragged = True
            self.is_placeholder = True
            try:
                self._batch_input_shape = tuple(input_tensor.shape.as_list())
            except ValueError:
                # If the shape cannot be represented as a tuple (e.g. unknown
                # rank)
                self._batch_input_shape = None
        elif input_tensor is None:
            if input_shape is not None:
                batch_input_shape = (batch_size,) + tuple(input_shape)
            else:
                batch_input_shape = None
            graph = backend.get_graph()
            with graph.as_default():
                input_tensor = backend.placeholder(
                    shape=batch_input_shape,
                    dtype=dtype,
                    name=self.name,
                    sparse=sparse,
                    ragged=ragged,
                )
            self.is_placeholder = True
            self._batch_input_shape = batch_input_shape
        else:
            if tf.compat.v1.executing_eagerly_outside_functions():
                if not isinstance(input_tensor, keras_tensor.KerasTensor):
                    input_tensor = keras_tensor.keras_tensor_from_tensor(
                        input_tensor
                    )
            else:
                if not tf_utils.is_symbolic_tensor(input_tensor):
                    raise ValueError(
                        "You should not pass an EagerTensor to `Input`. "
                        "For example, instead of creating an "
                        "`InputLayer`, you should instantiate your model "
                        "and directly call it on your input."
                    )
            self.is_placeholder = False
            try:
                self._batch_input_shape = tuple(input_tensor.shape.as_list())
            except ValueError:
                # If the shape cannot be represented as a tuple (e.g. unknown
                # rank)
                self._batch_input_shape = None
        # Create an input node.
        input_tensor._keras_mask = None
        node_module.Node(layer=self, outputs=input_tensor)
        # Store type spec
        if isinstance(input_tensor, keras_tensor.KerasTensor) or (
            tf_utils.is_extension_type(input_tensor)
        ):
            self._type_spec = input_tensor._type_spec
        else:
            self._type_spec = tf.TensorSpec(
                shape=input_tensor.shape,
                dtype=input_tensor.dtype,
                name=self.name,
            )
    def get_config(self):
        if self._init_type_spec is not None:
            config = {"name": self.name, "type_spec": self._init_type_spec}
        else:
            config = {
                "batch_input_shape": self._batch_input_shape,
                "dtype": self.dtype,
                "sparse": self.sparse,
                "ragged": self.ragged,
                "name": self.name,
            }
        return config
    @property
    def _trackable_saved_model_saver(self):
        return layer_serialization.InputLayerSavedModelSaver(self)
