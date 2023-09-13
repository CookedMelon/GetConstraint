"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.Orthogonal", "keras.initializers.orthogonal", v1=[]
)
class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.
    Also available via the shortcut function `tf.keras.initializers.orthogonal`.
    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.
    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Orthogonal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      gain: multiplicative factor to apply to the orthogonal matrix
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    References:
      - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """
    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to an orthogonal matrix.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(
            self.__class__.__name__, kwargs, support_partition=False
        )
        dtype = _assert_float_dtype(_get_dtype(dtype))
        # Check the shape
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be "
                "at least two-dimensional. Received: "
                f"shape={shape} of rank {len(shape)}."
            )
        self._warn_reuse()
        layout = kwargs.pop("layout", None)
        if layout:
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._generate_init_val, layout, shape=shape, dtype=dtype
            )
        return self._generate_init_val(shape, dtype)
    def _generate_init_val(self, shape, dtype):
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        # Generate a random matrix
        a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        # Compute the qr factorization
        q, r = tf.linalg.qr(a, full_matrices=False)
        # Make Q uniform
        d = tf.linalg.tensor_diag_part(r)
        q *= tf.sign(d)
        if num_rows < num_cols:
            q = tf.linalg.matrix_transpose(q)
        return self.gain * tf.reshape(q, shape)
    def get_config(self):
        return {"gain": self.gain, "seed": self.seed}
