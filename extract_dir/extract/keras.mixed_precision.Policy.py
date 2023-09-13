@keras_export("keras.mixed_precision.Policy", v1=[])
class Policy:
    """A dtype policy for a Keras layer.
    A dtype policy determines a layer's computation and variable dtypes. Each
    layer has a policy. Policies can be passed to the `dtype` argument of layer
    constructors, or a global policy can be set with
    `tf.keras.mixed_precision.set_global_policy`.
    Args:
      name: The policy name, which determines the compute and variable dtypes.
        Can be any dtype name, such as `'float32'` or `'float64'`, which causes
        both the compute and variable dtypes will be that dtype. Can also be the
        string `'mixed_float16'` or `'mixed_bfloat16'`, which causes the compute
        dtype to be float16 or bfloat16 and the variable dtype to be float32.
    Typically you only need to interact with dtype policies when using mixed
    precision, which is the use of float16 or bfloat16 for computations and
    float32 for variables. This is why the term `mixed_precision` appears in the
    API name. Mixed precision can be enabled by passing `'mixed_float16'` or
    `'mixed_bfloat16'` to `tf.keras.mixed_precision.set_global_policy`. See [the
    mixed precision
    guide](https://www.tensorflow.org/guide/keras/mixed_precision) for more
    information on how to use mixed precision.
    >>> tf.keras.mixed_precision.set_global_policy('mixed_float16')
    >>> layer1 = tf.keras.layers.Dense(10)
    >>> layer1.dtype_policy  # `layer1` will automatically use mixed precision
    <Policy "mixed_float16">
    >>> # Can optionally override layer to use float32
    >>> # instead of mixed precision.
    >>> layer2 = tf.keras.layers.Dense(10, dtype='float32')
    >>> layer2.dtype_policy
    <Policy "float32">
    >>> # Set policy back to initial float32 for future examples.
    >>> tf.keras.mixed_precision.set_global_policy('float32')
    In the example above, passing `dtype='float32'` to the layer is equivalent
    to passing `dtype=tf.keras.mixed_precision.Policy('float32')`. In general,
    passing a dtype policy name to a layer is equivalent to passing the
    corresponding policy, so it is never necessary to explicitly construct a
    `Policy` object.
    Note: `Model.compile` will automatically wrap an optimizer with a
    `tf.keras.mixed_precision.LossScaleOptimizer` if you use the
    `'mixed_float16'` policy. If you use a custom training loop instead of
    calling `Model.compile`, you should explicitly use a
    `tf.keras.mixed_precision.LossScaleOptimizer` to avoid numeric underflow
    with float16.
    ### How a layer uses its policy's compute dtype
    A layer casts its inputs to its compute dtype. This causes the layer's
    computations and output to also be in the compute dtype. For example:
    >>> x = tf.ones((4, 4, 4, 4), dtype='float64')
    >>> # `layer`'s policy defaults to float32.
    >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)
    >>> layer.compute_dtype  # Equivalent to layer.dtype_policy.compute_dtype
    'float32'
    >>> # `layer` casts its inputs to its compute dtype and does computations in
    >>> # that dtype.
    >>> y = layer(x)
    >>> y.dtype
    tf.float32
    Note that the base `tf.keras.layers.Layer` class inserts the casts. If
    subclassing your own layer, you do not have to insert any casts.
    Currently, only tensors in the first argument to the layer's `call` method
    are casted (although this will likely be changed in a future minor release).
    For example:
    >>> class MyLayer(tf.keras.layers.Layer):
    ...   # Bug! `b` will not be casted.
    ...   def call(self, a, b):
    ...     return a + 1., b + 1.
    >>> a = tf.constant(1., dtype="float32")
    >>> b = tf.constant(1., dtype="float32")
    >>> layer = MyLayer(dtype="float64")
    >>> x, y = layer(a, b)
    >>> x.dtype
    tf.float64
    >>> y.dtype
    tf.float32
    If writing your own layer with multiple inputs, you should either explicitly
    cast other tensors to `self.compute_dtype` in `call` or accept all tensors
    in the first argument as a list.
    The casting only occurs in TensorFlow 2. If
    `tf.compat.v1.disable_v2_behavior()` has been called, you can enable the
    casting behavior with
    `tf.compat.v1.keras.layers.enable_v2_dtype_behavior()`.
    ### How a layer uses its policy's variable dtype
    The default dtype of variables created by `tf.keras.layers.Layer.add_weight`
    is the layer's policy's variable dtype.
    If a layer's compute and variable dtypes differ, `add_weight` will wrap
    floating-point variables with a special wrapper called an
    `AutoCastVariable`.  `AutoCastVariable` is identical to the original
    variable except it casts itself to the layer's compute dtype when used
    within `Layer.call`. This means if you are writing a layer, you do not have
    to explicitly cast the variables to the layer's compute dtype. For example:
    >>> class SimpleDense(tf.keras.layers.Layer):
    ...
    ...   def build(self, input_shape):
    ...     # With mixed precision, self.kernel is a float32 AutoCastVariable
    ...     self.kernel = self.add_weight('kernel', (input_shape[-1], 10))
    ...
    ...   def call(self, inputs):
    ...     # With mixed precision, self.kernel will be casted to float16
    ...     return tf.linalg.matmul(inputs, self.kernel)
    ...
    >>> layer = SimpleDense(dtype='mixed_float16')
    >>> y = layer(tf.ones((10, 10)))
    >>> y.dtype
    tf.float16
    >>> layer.kernel.dtype
    tf.float32
    A layer author can prevent a variable from being wrapped with an
    `AutoCastVariable` by passing `experimental_autocast=False` to `add_weight`,
    which is useful if the float32 value of the variable must be accessed within
    the layer.
    ### How to write a layer that supports mixed precision and float64.
    For the most part, layers will automatically support mixed precision and
    float64 without any additional work, due to the fact the base layer
    automatically casts inputs, creates variables of the correct type, and in
    the case of mixed precision, wraps variables with `AutoCastVariables`.
    The primary case where you need extra work to support mixed precision or
    float64 is when you create a new tensor, such as with `tf.ones` or
    `tf.random.normal`, In such cases, you must create the tensor of the correct
    dtype. For example, if you call `tf.random.normal`, you must pass the
    compute dtype, which is the dtype the inputs have been casted to:
    >>> class AddRandom(tf.keras.layers.Layer):
    ...
    ...   def call(self, inputs):
    ...     # We must pass `dtype=inputs.dtype`, otherwise a TypeError may
    ...     # occur when adding `inputs` to `rand`.
    ...     rand = tf.random.normal(shape=inputs.shape, dtype=inputs.dtype)
    ...     return inputs + rand
    >>> layer = AddRandom(dtype='mixed_float16')
    >>> y = layer(x)
    >>> y.dtype
    tf.float16
    If you did not pass `dtype=inputs.dtype` to `tf.random.normal`, a
    `TypeError` would have occurred. This is because the `tf.random.normal`'s
    dtype defaults to `"float32"`, but the input dtype is float16. You cannot
    add a float32 tensor with a float16 tensor.
    """
    def __init__(self, name):
        if isinstance(name, tf.DType):
            raise TypeError(
                "'name' must be a string, not a DType. "
                f"Instead, pass DType.name. Received: name={name.name}"
            )
        elif not isinstance(name, str):
            raise TypeError(f"'name' must be a string, but got: {name}")
        self._name = name
        self._compute_dtype, self._variable_dtype = self._parse_name(name)
        if name in ("mixed_float16", "mixed_bloat16"):
            device_compatibility_check.log_device_compatibility_check(name)
    def _parse_name(self, name):
        """Parses a Policy name into a compute and variable dtype.
        Args:
          name: The name of the policy:
        Returns:
          The (compute_dtype, variable_dtype) pair.
        """
        if name.endswith("_float32_vars"):
            error_msg = (
                "Policies ending in '_float32_vars' have been removed "
                "from TensorFlow."
            )
            if name in ("infer_float32_vars", "infer_with_float32_vars"):
                error_msg += (
                    " Please use the 'mixed_float16' or 'mixed_bfloat16' "
                    "policy instead."
                )
            elif name == "float16_with_float32_vars":
                error_msg += " Please use the 'mixed_float16' policy instead."
            elif name == "bfloat16_with_float32_vars":
                error_msg += " Please use the 'mixed_bfloat16' policy instead."
            error_msg += f" Got policy name: '{name}'"
            raise ValueError(error_msg)
        if name == "mixed_float16":
            return "float16", "float32"
        elif name == "mixed_bfloat16":
            return "bfloat16", "float32"
        elif name == "_infer":
            # The "_infer" policy exists only for compatibility with TF 1, where
            # "_infer" is the default. The behavior matches the behavior of TF
            # 1's behavior before policies were introduced. With "_infer", the
            # computation and variable dtype are inferred from the first input
            # the first time the layer is called. Once the layer is called for
            # the first time, the layer's policy will change to the dtype of the
            # first input, and it will no longer have the "_infer" policy.
            #
            # The infer policy should be considered an implementation detail and
            # may be removed in the future.
            return None, None
        try:
            dtype = tf.as_dtype(name).name
        except TypeError:
            raise ValueError(
                f"Cannot convert value {name} to a mixed precision Policy. "
                "Valid policies include 'mixed_float16', 'mixed_bfloat16', "
                "and the name of any dtype such as 'float32'."
            )
        return dtype, dtype
    @property
    def variable_dtype(self):
        """The variable dtype of this policy.
        This is the dtype layers will create their variables in, unless a layer
        explicitly chooses a different dtype. If this is different than
        `Policy.compute_dtype`, Layers will cast variables to the compute dtype
        to avoid type errors.
        Variable regularizers are run in the variable dtype, not the compute
        dtype.
        Returns:
          The variable dtype of this policy, as a string.
        """
        return self._variable_dtype
    @property
    def compute_dtype(self):
        """The compute dtype of this policy.
        This is the dtype layers will do their computations in. Typically layers
        output tensors with the compute dtype as well.
        Note that even if the compute dtype is float16 or bfloat16, hardware
        devices may not do individual adds, multiplies, and other fundamental
        operations in float16 or bfloat16, but instead may do some of them in
        float32 for numeric stability. The compute dtype is the dtype of the
        inputs and outputs of the TensorFlow ops that the layer executes.
        Internally, many TensorFlow ops will do certain internal calculations in
        float32 or some other device-internal intermediate format with higher
        precision than float16/bfloat16, to increase numeric stability.
        For example, a `tf.keras.layers.Dense` layer, when run on a GPU with a
        float16 compute dtype, will pass float16 inputs to `tf.linalg.matmul`.
        But, `tf.linalg.matmul` will do use float32 intermediate math. The
        performance benefit of float16 is still apparent, due to increased
        memory bandwidth and the fact modern GPUs have specialized hardware for
        computing matmuls on float16 inputs while still keeping intermediate
        computations in float32.
        Returns:
          The compute dtype of this policy, as a string.
        """
        return self._compute_dtype
    @property
    def name(self):
        """Returns the name of this policy."""
        return self._name
    def __repr__(self):
        return f'<Policy "{self._name}">'
    def get_config(self):
        return {"name": self.name}
    @classmethod
    def from_config(cls, config, custom_objects=None):
        del custom_objects
        if "loss_scale" in config:
            config = config.copy()
            # Policy.get_config in TensorFlow 2.3 and below had a loss_scale. We
            # silently drop it.
            del config["loss_scale"]
        return cls(**config)
