"/home/cc/Workspace/tfconstraint/keras/saving/serialization_lib.py"
@keras_export(
    "keras.saving.serialize_keras_object", "keras.utils.serialize_keras_object"
)
def serialize_keras_object(obj):
    """Retrieve the config dict by serializing the Keras object.
    `serialize_keras_object()` serializes a Keras object to a python dictionary
    that represents the object, and is a reciprocal function of
    `deserialize_keras_object()`. See `deserialize_keras_object()` for more
    information about the config format.
    Args:
      obj: the Keras object to serialize.
    Returns:
      A python dict that represents the object. The python dict can be
      deserialized via `deserialize_keras_object()`.
    """
    # Fall back to legacy serialization for all TF1 users or if
    # wrapped by in_tf_saved_model_scope() to explicitly use legacy
    # saved_model logic.
    if not tf.__internal__.tf2.enabled() or in_tf_saved_model_scope():
        return legacy_serialization.serialize_keras_object(obj)
    if obj is None:
        return obj
    if isinstance(obj, PLAIN_TYPES):
        return obj
    if isinstance(obj, (list, tuple)):
        config_arr = [serialize_keras_object(x) for x in obj]
        return tuple(config_arr) if isinstance(obj, tuple) else config_arr
    if isinstance(obj, dict):
        return serialize_dict(obj)
    # Special cases:
    if isinstance(obj, bytes):
        return {
            "class_name": "__bytes__",
            "config": {"value": obj.decode("utf-8")},
        }
    if isinstance(obj, tf.TensorShape):
        return obj.as_list() if obj._dims is not None else None
    if isinstance(obj, tf.Tensor):
        return {
            "class_name": "__tensor__",
            "config": {
                "value": obj.numpy().tolist(),
                "dtype": obj.dtype.name,
            },
        }
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray) and obj.ndim > 0:
            return {
                "class_name": "__numpy__",
                "config": {
                    "value": obj.tolist(),
                    "dtype": obj.dtype.name,
                },
            }
        else:
            # Treat numpy floats / etc as plain types.
            return obj.item()
    if isinstance(obj, tf.DType):
        return obj.name
    if isinstance(obj, tf.compat.v1.Dimension):
        return obj.value
    if isinstance(obj, types.FunctionType) and obj.__name__ == "<lambda>":
        warnings.warn(
            "The object being serialized includes a `lambda`. This is unsafe. "
            "In order to reload the object, you will have to pass "
            "`safe_mode=False` to the loading function. "
            "Please avoid using `lambda` in the "
            "future, and use named Python functions instead. "
            f"This is the `lambda` being serialized: {inspect.getsource(obj)}",
            stacklevel=2,
        )
        return {
            "class_name": "__lambda__",
            "config": {
                "value": generic_utils.func_dump(obj),
            },
        }
    if isinstance(obj, tf.TypeSpec):
        ts_config = obj._serialize()
        # TensorShape and tf.DType conversion
        ts_config = list(
            map(
                lambda x: x.as_list()
                if isinstance(x, tf.TensorShape)
                else (x.name if isinstance(x, tf.DType) else x),
                ts_config,
            )
        )
        return {
            "class_name": "__typespec__",
            "spec_name": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "config": ts_config,
            "registered_name": None,
        }
    inner_config = _get_class_or_fn_config(obj)
    config_with_public_class = serialize_with_public_class(
        obj.__class__, inner_config
    )
    # TODO(nkovela): Add TF ops dispatch handler serialization for
    # ops.EagerTensor that contains nested numpy array.
    # Target: NetworkConstructionTest.test_constant_initializer_with_numpy
    if isinstance(inner_config, str) and inner_config == "op_dispatch_handler":
        return obj
    if config_with_public_class is not None:
        # Special case for non-serializable class modules
        if any(
            mod in config_with_public_class["module"]
            for mod in NON_SERIALIZABLE_CLASS_MODULES
        ):
            return obj
        get_build_and_compile_config(obj, config_with_public_class)
        record_object_after_serialization(obj, config_with_public_class)
        return config_with_public_class
    # Any custom object or otherwise non-exported object
    if isinstance(obj, types.FunctionType):
        module = obj.__module__
    else:
        module = obj.__class__.__module__
    class_name = obj.__class__.__name__
    if module == "builtins":
        registered_name = None
    else:
        if isinstance(obj, types.FunctionType):
            registered_name = object_registration.get_registered_name(obj)
        else:
            registered_name = object_registration.get_registered_name(
                obj.__class__
            )
    config = {
        "module": module,
        "class_name": class_name,
        "config": inner_config,
        "registered_name": registered_name,
    }
    get_build_and_compile_config(obj, config)
    record_object_after_serialization(obj, config)
    return config
def get_build_and_compile_config(obj, config):
    if hasattr(obj, "get_build_config"):
        build_config = obj.get_build_config()
        if build_config is not None:
            config["build_config"] = serialize_dict(build_config)
    if hasattr(obj, "get_compile_config"):
        compile_config = obj.get_compile_config()
        if compile_config is not None:
            config["compile_config"] = serialize_dict(compile_config)
    return
def serialize_with_public_class(cls, inner_config=None):
    """Serializes classes from public Keras API or object registration.
    Called to check and retrieve the config of any class that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`.
    """
    # This gets the `keras.*` exported name, such as "keras.optimizers.Adam".
    keras_api_name = tf_export.get_canonical_name_for_symbol(
        cls, api_name="keras"
    )
    # Case of custom or unknown class object
    if keras_api_name is None:
        registered_name = object_registration.get_registered_name(cls)
        if registered_name is None:
            return None
        # Return custom object config with corresponding registration name
        return {
            "module": cls.__module__,
            "class_name": cls.__name__,
            "config": inner_config,
            "registered_name": registered_name,
        }
    # Split the canonical Keras API name into a Keras module and class name.
    parts = keras_api_name.split(".")
    return {
        "module": ".".join(parts[:-1]),
        "class_name": parts[-1],
        "config": inner_config,
        "registered_name": None,
    }
def serialize_with_public_fn(fn, config, fn_module_name=None):
    """Serializes functions from public Keras API or object registration.
    Called to check and retrieve the config of any function that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`. If function's module name is
    already known, returns corresponding config.
    """
    if fn_module_name:
        return {
            "module": fn_module_name,
            "class_name": "function",
            "config": config,
            "registered_name": config,
        }
    keras_api_name = tf_export.get_canonical_name_for_symbol(
        fn, api_name="keras"
    )
    if keras_api_name:
        parts = keras_api_name.split(".")
        return {
            "module": ".".join(parts[:-1]),
            "class_name": "function",
            "config": config,
            "registered_name": config,
        }
    else:
        registered_name = object_registration.get_registered_name(fn)
        if not registered_name and not fn.__module__ == "builtins":
            return None
        return {
            "module": fn.__module__,
            "class_name": "function",
            "config": config,
            "registered_name": registered_name,
        }
def _get_class_or_fn_config(obj):
    """Return the object's config depending on its type."""
    # Functions / lambdas:
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    # All classes:
    if hasattr(obj, "get_config"):
        config = obj.get_config()
        if not isinstance(config, dict):
            raise TypeError(
                f"The `get_config()` method of {obj} should return "
                f"a dict. It returned: {config}"
            )
        return serialize_dict(config)
    elif hasattr(obj, "__name__"):
        return object_registration.get_registered_name(obj)
    else:
        raise TypeError(
            f"Cannot serialize object {obj} of type {type(obj)}. "
            "To be serializable, "
            "a class must implement the `get_config()` method."
        )
def serialize_dict(obj):
    return {key: serialize_keras_object(value) for key, value in obj.items()}
