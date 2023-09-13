@keras_export("keras.metrics.MeanMetricWrapper")
class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric.
    You could use this class to quickly build a mean metric from a function. The
    function needs to have the signature `fn(y_true, y_pred)` and return a
    per-sample loss array. `MeanMetricWrapper.result()` will return
    the average metric value across all samples seen so far.
    For example:
    ```python
    def accuracy(y_true, y_pred):
      return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
    accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=accuracy)
    keras_model.compile(..., metrics=accuracy_metric)
    ```
    Args:
      fn: The metric function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: Keyword arguments to pass on to `fn`.
    """
    @dtensor_utils.inject_mesh
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.
        `y_true` and `y_pred` should have the same shape.
        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
          sample_weight: Optional `sample_weight` acts as a
            coefficient for the metric. If a scalar is provided, then the metric
            is simply scaled by the given value. If `sample_weight` is a tensor
            of size `[batch_size]`, then the metric for each sample of the batch
            is rescaled by the corresponding element in the `sample_weight`
            vector. If the shape of `sample_weight` is `[batch_size, d0, ..
            dN-1]` (or can be broadcasted to this shape), then each metric
            element of `y_pred` is scaled by the corresponding value of
            `sample_weight`. (Note on `dN-1`: all metric functions reduce by 1
            dimension, usually the last axis (-1)).
        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        [
            y_true,
            y_pred,
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(  # noqa: E501
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
        ag_fn = tf.__internal__.autograph.tf_convert(
            self._fn, tf.__internal__.autograph.control_status_ctx()
        )
        matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
        mask = losses_utils.get_mask(matches)
        sample_weight = losses_utils.apply_valid_mask(
            matches, sample_weight, mask, self.reduction
        )
        return super().update_state(matches, sample_weight=sample_weight)
    def get_config(self):
        config = {
            k: backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
            for k, v in self._fn_kwargs.items()
        }
        if type(self) is MeanMetricWrapper:
            # Only include function argument when the object is a
            # MeanMetricWrapper and not a subclass.
            config["fn"] = self._fn
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @classmethod
    def from_config(cls, config):
        from keras.metrics import get
        # Note that while MeanMetricWrapper itself isn't public, objects of this
        # class may be created and added to the model by calling model.compile.
        fn = config.pop("fn", None)
        if cls is MeanMetricWrapper:
            return cls(get(fn), **config)
        return super(MeanMetricWrapper, cls).from_config(config)
