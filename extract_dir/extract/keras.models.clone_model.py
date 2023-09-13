@keras_export("keras.models.clone_model")
def clone_model(model, input_tensors=None, clone_function=None):
    """Clone a Functional or Sequential `Model` instance.
    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.
    Note that
    `clone_model` will not preserve the uniqueness of shared objects within the
    model (e.g. a single variable attached to two distinct layers will be
    restored as two separate variables).
    Args:
        model: Instance of `Model`
            (could be a Functional model or a Sequential model).
        input_tensors: optional list of input tensors or InputLayer objects
            to build the model upon. If not provided,
            new `Input` objects will be created.
        clone_function: Callable to be used to clone each layer in the target
            model (except `InputLayer` instances). It takes as argument the
            layer instance to be cloned, and returns the corresponding layer
            instance to be used in the model copy. If unspecified, this callable
            defaults to the following serialization/deserialization function:
            `lambda layer: layer.__class__.from_config(layer.get_config())`.
            By passing a custom callable, you can customize your copy of the
            model, e.g. by wrapping certain layers of interest (you might want
            to replace all `LSTM` instances with equivalent
            `Bidirectional(LSTM(...))` instances, for example).
    Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights. The cloned model may behave
      differently from the original model if a custom `clone_function`
      modifies the layer.
    Example:
    ```python
    # Create a test Sequential model.
    model = keras.Sequential([
        keras.Input(shape=(728,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    # Create a copy of the test model (with freshly initialized weights).
    new_model = clone_model(model)
    ```
    Note that subclassed models cannot be cloned, since their internal
    layer structure is not known. To achieve equivalent functionality
    as `clone_model` in the case of a subclassed model, simply make sure
    that the model class implements `get_config()`
    (and optionally `from_config()`), and call:
    ```python
    new_model = model.__class__.from_config(model.get_config())
    ```
    """
    with serialization.DisableSharedObjectScope():
        if isinstance(model, Sequential):
            return _clone_sequential_model(
                model, input_tensors=input_tensors, layer_fn=clone_function
            )
        if isinstance(model, functional.Functional):
            # If the get_config() method is the same as a regular Functional
            # model, we're safe to use _clone_functional_model (which relies
            # on a Functional constructor). In the case where the get_config
            # is custom, this may not necessarily work, but if clone_function
            # or input_tensors are passed, we attempt it anyway
            # in order to preserve backwards compatibility.
            if generic_utils.is_default(model.get_config) or (
                clone_function or input_tensors
            ):
                return _clone_functional_model(
                    model, input_tensors=input_tensors, layer_fn=clone_function
                )
        # Case of a custom model class
        if clone_function or input_tensors:
            raise ValueError(
                "Arguments clone_function and input_tensors "
                "are only supported for Sequential models "
                "or Functional models. Received model of "
                f"type '{model.__class__.__name__}', with "
                f"clone_function={clone_function} and "
                f"input_tensors={input_tensors}"
            )
        # Note that a custom object scope may be required in this case.
        return model.__class__.from_config(model.get_config())
