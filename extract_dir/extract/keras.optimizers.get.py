@keras_export("keras.optimizers.get")
def get(identifier, **kwargs):
    """Retrieves a Keras Optimizer instance.
    Args:
        identifier: Optimizer identifier, one of - String: name of an optimizer
          - Dictionary: configuration dictionary. - Keras Optimizer instance (it
          will be returned unchanged). - TensorFlow Optimizer instance (it will
          be wrapped as a Keras Optimizer).
    Returns:
        A Keras Optimizer instance.
    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    use_legacy_optimizer = kwargs.pop("use_legacy_optimizer", False)
    if kwargs:
        raise TypeError(f"Invalid keyword arguments: {kwargs}")
    if isinstance(
        identifier,
        (
            Optimizer,
            base_optimizer_legacy.OptimizerV2,
        ),
    ):
        return identifier
    elif isinstance(identifier, base_optimizer.Optimizer):
        if tf.__internal__.tf2.enabled() and not is_arm_mac():
            return identifier
        else:
            # If TF2 is disabled or on a M1 mac, we convert to the legacy
            # optimizer. We observed a slowdown of optimizer on M1 Mac, so we
            # fall back to the legacy optimizer for now, see b/263339144
            # for more context.
            optimizer_name = identifier.__class__.__name__
            logging.warning(
                "There is a known slowdown when using v2.11+ Keras optimizers "
                "on M1/M2 Macs. Falling back to the "
                "legacy Keras optimizer, i.e., "
                f"`tf.keras.optimizers.legacy.{optimizer_name}`."
            )
            return convert_to_legacy_optimizer(identifier)
    # Wrap legacy TF optimizer instances
    elif isinstance(identifier, tf.compat.v1.train.Optimizer):
        opt = TFOptimizer(identifier)
        backend.track_tf_optimizer(opt)
        return opt
    elif isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(
            identifier,
            use_legacy_optimizer=use_legacy_optimizer,
            use_legacy_format=use_legacy_format,
        )
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return get(
            config,
            use_legacy_optimizer=use_legacy_optimizer,
        )
    else:
        raise ValueError(
            f"Could not interpret optimizer identifier: {identifier}"
        )
