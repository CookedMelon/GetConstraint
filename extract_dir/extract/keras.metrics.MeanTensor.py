@keras_export("keras.metrics.MeanTensor")
class MeanTensor(Metric):
    """Computes the element-wise (weighted) mean of the given tensors.
    `MeanTensor` returns a tensor with the same shape of the input tensors. The
    mean value is updated by keeping local variables `total` and `count`. The
    `total` tracks the sum of the weighted values, and `count` stores the sum of
    the weighted counts.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      shape: (Optional) A list of integers, a tuple of integers, or a 1-D Tensor
        of type int32. If not specified, the shape is inferred from the values
        at the first call of update_state.
    Standalone usage:
    >>> m = tf.keras.metrics.MeanTensor()
    >>> m.update_state([0, 1, 2, 3])
    >>> m.update_state([4, 5, 6, 7])
    >>> m.result().numpy()
    array([2., 3., 4., 5.], dtype=float32)
    >>> m.update_state([12, 10, 8, 6], sample_weight= [0, 0.2, 0.5, 1])
    >>> m.result().numpy()
    array([2.       , 3.6363635, 4.8      , 5.3333335], dtype=float32)
    >>> m = tf.keras.metrics.MeanTensor(dtype=tf.float64, shape=(1, 4))
    >>> m.result().numpy()
    array([[0., 0., 0., 0.]])
    >>> m.update_state([[0, 1, 2, 3]])
    >>> m.update_state([[4, 5, 6, 7]])
    >>> m.result().numpy()
    array([[2., 3., 4., 5.]])
    """
    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_tensor", dtype=None, shape=None):
        super().__init__(name=name, dtype=dtype)
        self._shape = None
        self._total = None
        self._count = None
        self._built = False
        if shape is not None:
            self._build(shape)
    def _build(self, shape):
        self._shape = tf.TensorShape(shape)
        self._build_input_shape = self._shape
        # Create new state variables
        self._total = self.add_weight(
            name="total", shape=shape, initializer="zeros"
        )
        self._count = self.add_weight(
            name="count", shape=shape, initializer="zeros"
        )
        with tf.init_scope():
            if not tf.executing_eagerly():
                backend._initialize_variables(backend._get_session())
        self._built = True
    @property
    def total(self):
        return self._total if self._built else None
    @property
    def count(self):
        return self._count if self._built else None
    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the element-wise mean.
        Args:
          values: Per-example value.
          sample_weight: Optional weighting of each example. Defaults to `1`.
        Returns:
          Update op.
        """
        values = tf.cast(values, self._dtype)
        if not self._built:
            self._build(values.shape)
        elif values.shape != self._shape:
            raise ValueError(
                "MeanTensor input values must always have the same "
                "shape. Expected shape (set during the first call): "
                f"{self._shape}. "
                f"Got: {values.shape}."
            )
        num_values = tf.ones_like(values)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            # Update dimensions of weights to match with values if possible.
            (
                values,
                _,
                sample_weight,
            ) = losses_utils.squeeze_or_expand_dimensions(
                values, sample_weight=sample_weight
            )
            try:
                # Broadcast weights if possible.
                sample_weight = tf.__internal__.ops.broadcast_weights(
                    sample_weight, values
                )
            except ValueError:
                # Reduce values to same ndim as weight array
                ndim = backend.ndim(values)
                weight_ndim = backend.ndim(sample_weight)
                values = tf.reduce_mean(
                    values, axis=list(range(weight_ndim, ndim))
                )
            num_values = tf.multiply(num_values, sample_weight)
            values = tf.multiply(values, sample_weight)
        update_total_op = self._total.assign_add(values)
        with tf.control_dependencies([update_total_op]):
            return self._count.assign_add(num_values)
    def result(self):
        if not self._built:
            raise ValueError(
                "MeanTensor does not have any value yet. Please call the "
                "MeanTensor instance or use `.update_state(value)` "
                "before retrieving the result."
            )
        return tf.math.divide_no_nan(self.total, self.count)
    def reset_state(self):
        if self._built:
            backend.batch_set_value(
                [(v, np.zeros(v.shape.as_list())) for v in self.variables]
            )
