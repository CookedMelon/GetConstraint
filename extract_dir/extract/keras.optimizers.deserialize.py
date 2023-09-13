@keras_export("keras.optimizers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False, **kwargs):
    """Inverse of the `serialize` function.
    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.
    Returns:
        A Keras Optimizer instance.
    """
    # loss_scale_optimizer has a direct dependency of optimizer, import here
    # rather than top to avoid the cyclic dependency.
    from keras.mixed_precision import (
        loss_scale_optimizer,
    )
    use_legacy_optimizer = kwargs.pop("use_legacy_optimizer", False)
    if kwargs:
        raise TypeError(f"Invalid keyword arguments: {kwargs}")
    if len(config["config"]) > 0:
        # If the optimizer config is not empty, then we use the value of
        # `is_legacy_optimizer` to override `use_legacy_optimizer`. If
        # `is_legacy_optimizer` does not exist in config, it means we are
        # using the legacy optimzier.
        use_legacy_optimizer = config["config"].get("is_legacy_optimizer", True)
    if (
        tf.__internal__.tf2.enabled()
        and tf.executing_eagerly()
        and not is_arm_mac()
        and not use_legacy_optimizer
    ):
        # We observed a slowdown of optimizer on M1 Mac, so we fall back to the
        # legacy optimizer for M1 users now, see b/263339144 for more context.
        all_classes = {
            "adadelta": adadelta.Adadelta,
            "adagrad": adagrad.Adagrad,
            "adam": adam.Adam,
            "adamax": adamax.Adamax,
            "experimentaladadelta": adadelta.Adadelta,
            "experimentaladagrad": adagrad.Adagrad,
            "experimentaladam": adam.Adam,
            "experimentalsgd": sgd.SGD,
            "nadam": nadam.Nadam,
            "rmsprop": rmsprop.RMSprop,
            "sgd": sgd.SGD,
            "ftrl": ftrl.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizerV3,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }
    else:
        all_classes = {
            "adadelta": adadelta_legacy.Adadelta,
            "adagrad": adagrad_legacy.Adagrad,
            "adam": adam_legacy.Adam,
            "adamax": adamax_legacy.Adamax,
            "experimentaladadelta": adadelta.Adadelta,
            "experimentaladagrad": adagrad.Adagrad,
            "experimentaladam": adam.Adam,
            "experimentalsgd": sgd.SGD,
            "nadam": nadam_legacy.Nadam,
            "rmsprop": rmsprop_legacy.RMSprop,
            "sgd": gradient_descent_legacy.SGD,
            "ftrl": ftrl_legacy.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizer,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }
    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in all_classes:
        config["class_name"] = config["class_name"].lower()
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=all_classes,
            custom_objects=custom_objects,
            printable_module_name="optimizer",
        )
    return deserialize_keras_object(
        config,
        module_objects=all_classes,
        custom_objects=custom_objects,
        printable_module_name="optimizer",
    )
