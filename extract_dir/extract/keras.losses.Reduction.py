@keras_export("keras.losses.Reduction", v1=[])
class ReductionV2:
    """Types of loss reduction.
    Contains the following values:
    * `AUTO`: Indicates that the reduction option will be determined by the
      usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, we expect
      reduction value to be `SUM` or `NONE`. Using `AUTO` in that case will
      raise an error.
    * `NONE`: No **additional** reduction is applied to the output of the
      wrapped loss function. When non-scalar losses are returned to Keras
      functions like `fit`/`evaluate`, the unreduced vector loss is passed to
      the optimizer but the reported loss will be a scalar value.
       Caution: **Verify the shape of the outputs when using** `Reduction.NONE`.
       The builtin loss functions wrapped by the loss classes reduce one
       dimension (`axis=-1`, or `axis` if specified by loss function).
       `Reduction.NONE` just means that no **additional** reduction is applied
       by the class wrapper. For categorical losses with an example input shape
       of `[batch, W, H, n_classes]` the `n_classes` dimension is reduced. For
       pointwise losses you must include a dummy axis so that `[batch, W, H, 1]`
       is reduced to `[batch, W, H]`. Without the dummy axis `[batch, W, H]`
       will be incorrectly reduced to `[batch, W]`.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in
       losses.  This reduction type is not supported when used with
       `tf.distribute.Strategy` outside of built-in training loops like
       `tf.keras` `compile`/`fit`.
       You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
       ```
       with strategy.scope():
         loss_obj = tf.keras.losses.CategoricalCrossentropy(
             reduction=tf.keras.losses.Reduction.NONE)
         ....
         loss = tf.reduce_sum(loss_obj(labels, predictions)) *
             (1. / global_batch_size)
       ```
    Please see the [custom training guide](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
    details on this.
    """
    AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"
    @classmethod
    def all(cls):
        return (cls.AUTO, cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)
    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError(
                f'Invalid Reduction Key: {key}. Expected keys are "{cls.all()}"'
            )
