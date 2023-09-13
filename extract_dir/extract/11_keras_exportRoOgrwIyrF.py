"/home/cc/Workspace/tfconstraint/keras/optimizers/__init__.py"
@keras_export(
    "keras.__internal__.optimizers.convert_to_legacy_optimizer", v1=[]
)
def convert_to_legacy_optimizer(optimizer):
    """Convert experimental optimizer to legacy optimizer.
    This function takes in a `keras.optimizers.Optimizer`
    instance and converts it to the corresponding
    `keras.optimizers.legacy.Optimizer` instance.
    For example, `keras.optimizers.Adam(...)` to
    `keras.optimizers.legacy.Adam(...)`.
    Args:
        optimizer: An instance of `keras.optimizers.Optimizer`.
    """
    # loss_scale_optimizer has a direct dependency of optimizer, import here
    # rather than top to avoid the cyclic dependency.
    from keras.mixed_precision import (
        loss_scale_optimizer,
    )
    if not isinstance(optimizer, base_optimizer.Optimizer):
        raise ValueError(
            "`convert_to_legacy_optimizer` should only be called "
            "on instances of `tf.keras.optimizers.Optimizer`, but "
            f"received {optimizer} of type {type(optimizer)}."
        )
    optimizer_name = optimizer.__class__.__name__.lower()
    config = optimizer.get_config()
    # Remove fields that only exist in experimental optimizer.
    keys_to_remove = [
        "weight_decay",
        "use_ema",
        "ema_momentum",
        "ema_overwrite_frequency",
        "jit_compile",
        "is_legacy_optimizer",
    ]
    for key in keys_to_remove:
        config.pop(key, None)
    if isinstance(optimizer, loss_scale_optimizer.LossScaleOptimizerV3):
        # For LossScaleOptimizers, recursively convert the inner optimizer
        config["inner_optimizer"] = convert_to_legacy_optimizer(
            optimizer.inner_optimizer
        )
        if optimizer_name == "lossscaleoptimizerv3":
            optimizer_name = "lossscaleoptimizer"
    # Learning rate can be a custom LearningRateSchedule, which is stored as
    # a dict in config, and cannot be deserialized.
    if hasattr(optimizer, "_learning_rate") and isinstance(
        optimizer._learning_rate, learning_rate_schedule.LearningRateSchedule
    ):
        config["learning_rate"] = optimizer._learning_rate
    legacy_optimizer_config = {
        "class_name": optimizer_name,
        "config": config,
    }
    return deserialize(legacy_optimizer_config, use_legacy_optimizer=True)
