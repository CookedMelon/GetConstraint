"/home/cc/Workspace/tfconstraint/keras/saving/serialization_lib.py"
@keras_export(
    "keras.saving.deserialize_keras_object",
    "keras.utils.deserialize_keras_object",
)
def deserialize_keras_object(
    config, custom_objects=None, safe_mode=True, **kwargs
):
    """Retrieve the object by deserializing the config dict.
    The config dict is a Python dictionary that consists of a set of key-value
    pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,
    `Metrics`, etc. The saving and loading library uses the following keys to
    record information of a Keras object:
    - `class_name`: String. This is the name of the class,
      as exactly defined in the source
      code, such as "LossesContainer".
    - `config`: Dict. Library-defined or user-defined key-value pairs that store
      the configuration of the object, as obtained by `object.get_config()`.
    - `module`: String. The path of the python module, such as
      "keras.engine.compile_utils". Built-in Keras classes
      expect to have prefix `keras`.
    - `registered_name`: String. The key the class is registered under via
      `keras.saving.register_keras_serializable(package, name)` API. The key has
      the format of '{package}>{name}', where `package` and `name` are the
      arguments passed to `register_keras_serializable()`. If `name` is not
      provided, it uses the class name. If `registered_name` successfully
      resolves to a class (that was registered), the `class_name` and `config`
      values in the dict will not be used. `registered_name` is only used for
      non-built-in classes.
    For example, the following dictionary represents the built-in Adam optimizer
    with the relevant config:
    ```python
    dict_structure = {
        "class_name": "Adam",
        "config": {
            "amsgrad": false,
            "beta_1": 0.8999999761581421,
            "beta_2": 0.9990000128746033,
            "decay": 0.0,
            "epsilon": 1e-07,
            "learning_rate": 0.0010000000474974513,
            "name": "Adam"
        },
        "module": "keras.optimizers",
        "registered_name": None
    }
    # Returns an `Adam` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```
    If the class does not have an exported Keras namespace, the library tracks
    it by its `module` and `class_name`. For example:
    ```python
    dict_structure = {
      "class_name": "LossesContainer",
      "config": {
          "losses": [...],
          "total_loss_mean": {...},
      },
      "module": "keras.engine.compile_utils",
      "registered_name": "LossesContainer"
    }
    # Returns a `LossesContainer` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```
    And the following dictionary represents a user-customized `MeanSquaredError`
    loss:
    ```python
    @keras.saving.register_keras_serializable(package='my_package')
    class ModifiedMeanSquaredError(keras.losses.MeanSquaredError):
      ...
    dict_structure = {
        "class_name": "ModifiedMeanSquaredError",
        "config": {
            "fn": "mean_squared_error",
            "name": "mean_squared_error",
            "reduction": "auto"
        },
        "registered_name": "my_package>ModifiedMeanSquaredError"
    }
    # Returns the `ModifiedMeanSquaredError` object
    deserialize_keras_object(dict_structure)
    ```
    Args:
        config: Python dict describing the object.
        custom_objects: Python dict containing a mapping between custom
            object names the corresponding classes or functions.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.
            When `safe_mode=False`, loading an object has the potential to
            trigger arbitrary code execution. This argument is only
            applicable to the Keras v3 model format. Defaults to `True`.
    Returns:
      The object described by the `config` dictionary.
    """
    safe_scope_arg = in_safe_mode()  # Enforces SafeModeScope
    safe_mode = safe_scope_arg if safe_scope_arg is not None else safe_mode
    module_objects = kwargs.pop("module_objects", None)
    custom_objects = custom_objects or {}
    tlco = object_registration._THREAD_LOCAL_CUSTOM_OBJECTS.__dict__
    gco = object_registration._GLOBAL_CUSTOM_OBJECTS
    custom_objects = {**custom_objects, **tlco, **gco}
    # Optional deprecated argument for legacy deserialization call
    printable_module_name = kwargs.pop("printable_module_name", "object")
    if kwargs:
        raise ValueError(
            "The following argument(s) are not supported: "
            f"{list(kwargs.keys())}"
        )
    # Fall back to legacy deserialization for all TF1 users or if
    # wrapped by in_tf_saved_model_scope() to explicitly use legacy
    # saved_model logic.
    if not tf.__internal__.tf2.enabled() or in_tf_saved_model_scope():
        return legacy_serialization.deserialize_keras_object(
            config, module_objects, custom_objects, printable_module_name
        )
    if config is None:
        return None
    if (
        isinstance(config, str)
        and custom_objects
        and custom_objects.get(config) is not None
    ):
        # This is to deserialize plain functions which are serialized as
        # string names by legacy saving formats.
        return custom_objects[config]
    if isinstance(config, (list, tuple)):
        return [
            deserialize_keras_object(
                x, custom_objects=custom_objects, safe_mode=safe_mode
            )
            for x in config
        ]
    if module_objects is not None:
        inner_config, fn_module_name, has_custom_object = None, None, False
        if isinstance(config, dict):
            if "config" in config:
                inner_config = config["config"]
            if "class_name" not in config:
                raise ValueError(
                    f"Unknown `config` as a `dict`, config={config}"
                )
            # Check case where config is function or class and in custom objects
            if custom_objects and (
                config["class_name"] in custom_objects
                or config.get("registered_name") in custom_objects
                or (
                    isinstance(inner_config, str)
                    and inner_config in custom_objects
                )
            ):
                has_custom_object = True
            # Case where config is function but not in custom objects
            elif config["class_name"] == "function":
                fn_module_name = config["module"]
                if fn_module_name == "builtins":
                    config = config["config"]
                else:
                    config = config["registered_name"]
            # Case where config is class but not in custom objects
            else:
                if config.get("module", "_") is None:
                    raise TypeError(
                        "Cannot deserialize object of type "
                        f"`{config['class_name']}`. If "
                        f"`{config['class_name']}` is a custom class, please "
                        "register it using the "
                        "`@keras.saving.register_keras_serializable()` "
                        "decorator."
                    )
                config = config["class_name"]
        if not has_custom_object:
            # Return if not found in either module objects or custom objects
            if config not in module_objects:
                # Object has already been deserialized
                return config
            if isinstance(module_objects[config], types.FunctionType):
                return deserialize_keras_object(
                    serialize_with_public_fn(
                        module_objects[config], config, fn_module_name
                    ),
                    custom_objects=custom_objects,
                )
            return deserialize_keras_object(
                serialize_with_public_class(
                    module_objects[config], inner_config=inner_config
                ),
                custom_objects=custom_objects,
            )
    if isinstance(config, PLAIN_TYPES):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"Could not parse config: {config}")
    if "class_name" not in config or "config" not in config:
        return {
            key: deserialize_keras_object(
                value, custom_objects=custom_objects, safe_mode=safe_mode
            )
            for key, value in config.items()
        }
    class_name = config["class_name"]
    inner_config = config["config"] or {}
    custom_objects = custom_objects or {}
    # Special cases:
    if class_name == "__tensor__":
        return tf.constant(inner_config["value"], dtype=inner_config["dtype"])
    if class_name == "__numpy__":
        return np.array(inner_config["value"], dtype=inner_config["dtype"])
    if config["class_name"] == "__bytes__":
        return inner_config["value"].encode("utf-8")
    if config["class_name"] == "__lambda__":
        if safe_mode:
            raise ValueError(
                "Requested the deserialization of a `lambda` object. "
                "This carries a potential risk of arbitrary code execution "
                "and thus it is disallowed by default. If you trust the "
                "source of the saved model, you can pass `safe_mode=False` to "
                "the loading function in order to allow `lambda` loading."
            )
        return generic_utils.func_load(inner_config["value"])
    if config["class_name"] == "__typespec__":
        obj = _retrieve_class_or_fn(
            config["spec_name"],
            config["registered_name"],
            config["module"],
            obj_type="class",
            full_config=config,
            custom_objects=custom_objects,
        )
        # Conversion to TensorShape and tf.DType
        inner_config = map(
            lambda x: tf.TensorShape(x)
            if isinstance(x, list)
            else (getattr(tf, x) if hasattr(tf.dtypes, str(x)) else x),
            inner_config,
        )
        return obj._deserialize(tuple(inner_config))
    # Below: classes and functions.
    module = config.get("module", None)
    registered_name = config.get("registered_name", class_name)
    if class_name == "function":
        fn_name = inner_config
        return _retrieve_class_or_fn(
            fn_name,
            registered_name,
            module,
            obj_type="function",
            full_config=config,
            custom_objects=custom_objects,
        )
    # Below, handling of all classes.
    # First, is it a shared object?
    if "shared_object_id" in config:
        obj = get_shared_object(config["shared_object_id"])
        if obj is not None:
            return obj
    cls = _retrieve_class_or_fn(
        class_name,
        registered_name,
        module,
        obj_type="class",
        full_config=config,
        custom_objects=custom_objects,
    )
    if isinstance(cls, types.FunctionType):
        return cls
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Unable to reconstruct an instance of '{class_name}' because "
            f"the class is missing a `from_config()` method. "
            f"Full object config: {config}"
        )
    # Instantiate the class from its config inside a custom object scope
    # so that we can catch any custom objects that the config refers to.
    custom_obj_scope = object_registration.custom_object_scope(custom_objects)
    safe_mode_scope = SafeModeScope(safe_mode)
    with custom_obj_scope, safe_mode_scope:
        instance = cls.from_config(inner_config)
        build_config = config.get("build_config", None)
        if build_config:
            instance.build_from_config(build_config)
        compile_config = config.get("compile_config", None)
        if compile_config:
            instance.compile_from_config(compile_config)
    if "shared_object_id" in config:
        record_object_after_deserialization(
            instance, config["shared_object_id"]
        )
    return instance
def _retrieve_class_or_fn(
    name, registered_name, module, obj_type, full_config, custom_objects=None
):
    # If there is a custom object registered via
    # `register_keras_serializable()`, that takes precedence.
    if obj_type == "function":
        custom_obj = object_registration.get_registered_object(
            name, custom_objects=custom_objects
        )
    else:
        custom_obj = object_registration.get_registered_object(
            registered_name, custom_objects=custom_objects
        )
    if custom_obj is not None:
        return custom_obj
    if module:
        # If it's a Keras built-in object,
        # we cannot always use direct import, because the exported
        # module name might not match the package structure
        # (e.g. experimental symbols).
        if module == "keras" or module.startswith("keras."):
            api_name = module + "." + name
            # Legacy internal APIs are stored in TF API naming dict
            # with `compat.v1` prefix
            if "__internal__.legacy" in api_name:
                api_name = "compat.v1." + api_name
            obj = tf_export.get_symbol_from_name(api_name)
            if obj is not None:
                return obj
        # Configs of Keras built-in functions do not contain identifying
        # information other than their name (e.g. 'acc' or 'tanh'). This special
        # case searches the Keras modules that contain built-ins to retrieve
        # the corresponding function from the identifying string.
        if obj_type == "function" and module == "builtins":
            for mod in BUILTIN_MODULES:
                obj = tf_export.get_symbol_from_name(
                    "keras." + mod + "." + name
                )
                if obj is not None:
                    return obj
            # Retrieval of registered custom function in a package
            filtered_dict = {
                k: v
                for k, v in custom_objects.items()
                if k.endswith(full_config["config"])
            }
            if filtered_dict:
                return next(iter(filtered_dict.values()))
        # Otherwise, attempt to retrieve the class object given the `module`
        # and `class_name`. Import the module, find the class.
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            raise TypeError(
                f"Could not deserialize {obj_type} '{name}' because "
                f"its parent module {module} cannot be imported. "
                f"Full object config: {full_config}"
            )
        obj = vars(mod).get(name, None)
        # Special case for keras.metrics.metrics
        if obj is None and registered_name is not None:
            obj = vars(mod).get(registered_name, None)
        if obj is not None:
            return obj
    raise TypeError(
        f"Could not locate {obj_type} '{name}'. "
        "Make sure custom classes are decorated with "
        "`@keras.saving.register_keras_serializable()`. "
        f"Full object config: {full_config}"
    )
