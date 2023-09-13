@keras_export("keras.mixed_precision.LossScaleOptimizer")
class BaseLossScaleOptimizer(metaclass=LossScaleOptimizerMetaclass):
    """An optimizer that applies loss scaling to prevent numeric underflow.
    Loss scaling is a technique to prevent numeric underflow in intermediate
    gradients when float16 is used. To prevent underflow, the loss is multiplied
    (or "scaled") by a certain factor called the "loss scale", which causes
    intermediate gradients to be scaled by the loss scale as well. The final
    gradients are divided (or "unscaled") by the loss scale to bring them back
    to their original value.
    `LossScaleOptimizer` wraps another optimizer and applies loss scaling to it.
    By default, the loss scale is dynamically updated over time so you do not
    have to choose the loss scale. The `minimize` method automatically scales
    the loss, unscales the gradients, and updates the loss scale so all you have
    to do is wrap your optimizer with a `LossScaleOptimizer` if you use
    `minimize`. For example:
    >>> opt = tf.keras.optimizers.experimental.SGD(0.25)
    >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    >>> var = tf.Variable(1.)
    >>> loss_fn = lambda: var ** 2
    >>> # 'minimize' applies loss scaling and updates the loss sale.
    >>> opt.minimize(loss_fn, var_list=[var])
    >>> var.numpy()
    0.5
    If a `tf.GradientTape` is used to compute gradients instead of `minimize`,
    you must scale the loss and gradients manually. This can be done with the
    `LossScaleOptimizer.get_scaled_loss` and
    `LossScaleOptimizer.get_unscaled_gradients` methods. For example:
    >>> with tf.GradientTape() as tape:
    ...   loss = loss_fn()
    ...   scaled_loss = opt.get_scaled_loss(loss)
    >>> scaled_grad = tape.gradient(scaled_loss, var)
    >>> (grad,) = opt.get_unscaled_gradients([scaled_grad])
    >>> opt.apply_gradients([(grad, var)])  # Loss scale is updated here
    >>> var.numpy()
    0.25
    Warning: If you forget to call `get_scaled_loss` or `get_unscaled_gradients`
    (or both) when using a `tf.GradientTape`, the model will likely converge to
    a worse quality. Please make sure you call each function exactly once.
    When mixed precision with float16 is used, there is typically no risk of
    underflow affecting model quality if loss scaling is properly used. See
    [the mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for more information
    on how to use mixed precision.
    Args:
      inner_optimizer: The `tf.keras.optimizers.Optimizer` or
        `tf.keras.optimizers.experimental.Optimizer` instance to wrap.
      dynamic: Bool indicating whether dynamic loss scaling is used. Defaults to
        True. If True, the loss scale will be dynamically updated over time
        using an algorithm that keeps the loss scale at approximately its
        optimal value.  If False, a single fixed loss scale is used and
        `initial_scale` must be specified, which is used as the loss scale.
        Recommended to keep as True, as choosing a fixed loss scale can be
        tricky. Currently, there is a small performance overhead to dynamic loss
        scaling compared to fixed loss scaling.
      initial_scale: The initial loss scale. If `dynamic` is True, this defaults
        to `2 ** 15`. If `dynamic` is False, this must be specified and acts as
        the sole loss scale, as the loss scale does not change over time. When
        dynamic loss scaling is used, is better for this to be a very high
        number, because a loss scale that is too high gets lowered far more
        quickly than a loss scale that is too low gets raised.
      dynamic_growth_steps: With dynamic loss scaling, every
        `dynamic_growth_steps` steps with finite gradients, the loss scale is
        doubled. Defaults to 2000. If a nonfinite gradient is encountered, the
        count is reset back to zero, gradients are skipped that step, and the
        loss scale is halved. The count can be queried with
        `LossScaleOptimizer.dynamic_counter`. This argument can only be
        specified if `dynamic` is True.
    `LossScaleOptimizer` will occasionally skip applying gradients to the
    variables, in which case the trainable variables will not change that step.
    This is done because the dynamic loss scale will sometimes be raised too
    high, causing overflow in the gradients. Typically, the first 2 to 15 steps
    of the model are skipped as the initial loss scale is very high, but
    afterwards steps will only be skipped on average 0.05% of the time (the
    fraction of steps skipped is `1 / dynamic_growth_steps`).
    `LossScaleOptimizer` delegates all public `Optimizer` methods to the inner
    optimizer. Additionally, in methods `minimize` and `get_gradients`, it
    scales the loss and unscales the gradients. In methods `minimize` and
    `apply_gradients`, it additionally updates the loss scale and skips applying
    gradients if any gradient has a nonfinite value.
    ### Hyperparameters
    If wrapping a `tf.keras.optimizers.Optimizer`, hyperparameters can be
    accessed and set on the LossScaleOptimizer, which will be delegated to the
    wrapped optimizer.
    >>> opt = tf.keras.optimizers.legacy.Adam(beta_1=0.8, epsilon=1e-5)
    >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    >>> opt.beta_1  # Equivalent to `opt.inner_optimizer.beta_1`
    0.8
    >>> opt.beta_1 = 0.7  # Equivalent to `opt.inner_optimizer.beta_1 = 0.7`
    >>> opt.beta_1
    0.7
    >>> opt.inner_optimizer.beta_1
    0.7
    However, accessing or setting non-hyperparameters is not delegated to the
    LossScaleOptimizer. In an Adam optimizer, `beta_1` is a hyperparameter but
    `epsilon` is not, as the Adam optimizer only calls `Optimizer._set_hyper` on
    `beta_1`.
    >>> opt.inner_optimizer.epsilon
    1e-5
    >>> opt.epsilon
    Traceback (most recent call last):
    ...
    AttributeError: 'LossScaleOptimizer' object has no attribute 'epsilon'
    >>> opt.epsilon = 1e-4  # This does NOT set epsilon on `opt.inner_optimizer`
    >>> opt.inner_optimizer.epsilon
    >>> 1e-5
    In the above example, despite epsilon being set on the LossScaleOptimizer,
    the old epsilon value will still be used when training as epsilon was not
    set on the inner optimizer.
    """
    @property
    def dynamic(self):
        """Bool indicating whether dynamic loss scaling is used."""
        raise NotImplementedError
    @property
    def loss_scale(self):
        """The current loss scale as a float32 scalar tensor."""
        raise NotImplementedError
    @property
    def dynamic_counter(self):
        """The number of steps since the loss scale was last increased or
        decreased.
        This is None if `LossScaleOptimizer.dynamic` is False.
        The counter is incremented every step. Once it reaches
        `LossScaleOptimizer.dynamic_growth_steps`, the loss scale will be
        doubled and the counter will be reset back to zero. If nonfinite
        gradients are encountered, the loss scale will be halved and the counter
        will be reset back to zero.
        """
        raise NotImplementedError
    @property
    def initial_scale(self):
        """The initial loss scale.
        If `LossScaleOptimizer.dynamic` is False, this is the same number as
        `LossScaleOptimizer.loss_scale`, as the loss scale never changes.
        """
        raise NotImplementedError
    @property
    def dynamic_growth_steps(self):
        """The number of steps it takes to increase the loss scale.
        This is None if `LossScaleOptimizer.dynamic` is False.
        Every `dynamic_growth_steps` consecutive steps with finite gradients,
        the loss scale is increased.
        """
        raise NotImplementedError
    @property
    def inner_optimizer(self):
        """The optimizer that this LossScaleOptimizer is wrapping."""
        raise NotImplementedError
    def get_scaled_loss(self, loss):
        """Scales the loss by the loss scale.
        This method is only needed if you compute gradients manually, e.g. with
        `tf.GradientTape`. In that case, call this method to scale the loss
        before passing the loss to `tf.GradientTape`. If you use
        `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`,
        loss scaling is automatically applied and this method is unneeded.
        If this method is called, `get_unscaled_gradients` should also be
        called.  See the `tf.keras.mixed_precision.LossScaleOptimizer` doc for
        an example.
        Args:
          loss: The loss, which will be multiplied by the loss scale. Can either
            be a tensor or a callable returning a tensor.
        Returns:
          `loss` multiplied by `LossScaleOptimizer.loss_scale`.
        """
        # Calls to this function would be delegated to `get_scaled_loss`
        # of either `LossScaleOptimizer` or `LossScaleOptimizerV3`, depending on
        # the type of `inner_optimizer`.
        raise NotImplementedError
    def get_unscaled_gradients(self, grads):
        """Unscales the gradients by the loss scale.
        This method is only needed if you compute gradients manually, e.g. with
        `tf.GradientTape`. In that case, call this method to unscale the
        gradients after computing them with `tf.GradientTape`. If you use
        `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`,
        loss scaling is automatically applied and this method is unneeded.
        If this method is called, `get_scaled_loss` should also be called. See
        the `tf.keras.mixed_precision.LossScaleOptimizer` doc for an
        example.
        Args:
          grads: A list of tensors, each which will be divided by the loss
            scale. Can have None values, which are ignored.
        Returns:
          A new list the same size as `grads`, where every non-None value in
          `grads` is divided by `LossScaleOptimizer.loss_scale`.
        """
        # Calls to this function would be delegated to `get_unscaled_gradients`
        # of either `LossScaleOptimizer` or `LossScaleOptimizerV3`, depending on
        # the type of `inner_optimizer`.
        raise NotImplementedError
