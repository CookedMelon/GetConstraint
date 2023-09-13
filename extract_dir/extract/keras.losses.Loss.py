@keras_export("keras.losses.Loss")
class Loss:
    """Loss base class.
    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`,
      `y_pred`.
    Example subclass implementation:
    ```python
    class MeanSquaredError(Loss):
      def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
    ```
    When using a Loss under a `tf.distribute.Strategy`, except passing it
    to `Model.compile()` for use by `Model.fit()`, please use reduction
    types 'SUM' or 'NONE', and reduce losses explicitly. Using 'AUTO' or
    'SUM_OVER_BATCH_SIZE' will raise an error when calling the Loss object
    from a custom training loop or from user-defined code in `Layer.call()`.
    Please see this custom training
    [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
    for more details on this.
    """
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """Initializes `Loss` class.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
              for more details.
          name: Optional name for the instance.
        """
        losses_utils.ReductionV2.validate(reduction)
        self.reduction = reduction
        self.name = name
        # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
        # CannedEstimators.
        self._allow_sum_over_batch_size = False
        self._set_name_scope()
    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            self._name_scope = self.__class__.__name__.strip("_")
        elif self.name == "<lambda>":
            self._name_scope = "lambda"
        else:
            # E.g. '_my_loss' => 'my_loss'
            self._name_scope = self.name.strip("_")
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Invokes the `Loss` instance.
        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
          sample_weight: Optional `sample_weight` acts as a coefficient for the
            loss. If a scalar is provided, then the loss is simply scaled by the
            given value. If `sample_weight` is a tensor of size `[batch_size]`,
            then the total loss for each sample of the batch is rescaled by the
            corresponding element in the `sample_weight` vector. If the shape of
            `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
            broadcasted to this shape), then each loss element of `y_pred` is
            scaled by the corresponding value of `sample_weight`. (Note
            on`dN-1`: all loss functions reduce by 1 dimension, usually
            axis=-1.)
        Returns:
          Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
            shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note
            `dN-1` because all loss functions reduce by 1 dimension, usually
            axis=-1.)
        Raises:
          ValueError: If the shape of `sample_weight` is invalid.
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is
        # not accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight
        )
        with backend.name_scope(self._name_scope), graph_ctx:
            if tf.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = tf.__internal__.autograph.tf_convert(
                    self.call, tf.__internal__.autograph.control_status_ctx()
                )
            losses = call_fn(y_true, y_pred)
            in_mask = losses_utils.get_mask(y_pred)
            out_mask = losses_utils.get_mask(losses)
            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None
            reduction = self._get_reduction()
            sample_weight = losses_utils.apply_valid_mask(
                losses, sample_weight, mask, reduction
            )
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=reduction
            )
    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).
        Args:
            config: Output of `get_config()`.
        Returns:
            A `Loss` instance.
        """
        return cls(**config)
    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        return {"reduction": self.reduction, "name": self.name}
    @abc.abstractmethod
    @doc_controls.for_subclass_implementers
    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.
        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
        Returns:
          Loss values with the shape `[batch_size, d0, .. dN-1]`.
        """
        raise NotImplementedError("Must be implemented in subclasses.")
    def _get_reduction(self):
        """Handles `AUTO` reduction cases and returns the reduction value."""
        if (
            not self._allow_sum_over_batch_size
            and tf.distribute.has_strategy()
            and (
                self.reduction == losses_utils.ReductionV2.AUTO
                or self.reduction
                == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
            )
        ):
            raise ValueError(
                "Please use `tf.keras.losses.Reduction.SUM` or "
                "`tf.keras.losses.Reduction.NONE` for loss reduction when "
                "losses are used with `tf.distribute.Strategy`, "
                "except for specifying losses in `Model.compile()` "
                "for use by the built-in training looop `Model.fit()`.\n"
                "Please see https://www.tensorflow.org/tutorials"
                "/distribute/custom_training for more details."
            )
        if self.reduction == losses_utils.ReductionV2.AUTO:
            return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        return self.reduction
