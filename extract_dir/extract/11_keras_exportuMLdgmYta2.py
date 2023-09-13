"/home/cc/Workspace/tfconstraint/python/keras/models.py"
@keras_export(
    'keras.__internal__.models.in_place_subclassed_model_state_restoration',
    v1=[])
def in_place_subclassed_model_state_restoration(model):
  """Restores the original state of a model after it was "reset".
  This undoes this action of `_in_place_subclassed_model_reset`, which is called
  in `clone_and_build_model` if `in_place_reset` is set to True.
  Args:
    model: Instance of a Keras model created via subclassing, on which
      `_in_place_subclassed_model_reset` was previously called.
  """
  assert not model._is_graph_network
  # Restore layers and build attributes
  if (hasattr(model, '_original_attributes_cache') and
      model._original_attributes_cache is not None):
    # Models have sticky attribute assignment, so we want to be careful to add
    # back the previous attributes and track Layers by their original names
    # without adding dependencies on "utility" attributes which Models exempt
    # when they're constructed.
    setattr_tracking = model._setattr_tracking
    model._setattr_tracking = False
    model._self_tracked_trackables = []
    for name, value in model._original_attributes_cache.items():
      setattr(model, name, value)
      if isinstance(value, Layer):
        model._self_tracked_trackables.append(value)
    model._original_attributes_cache = None
    model._setattr_tracking = setattr_tracking
  else:
    # Restore to the state of a never-called model.
    _reset_build_compile_trackers(model)
@keras_export('keras.__internal__.models.clone_and_build_model', v1=[])
def clone_and_build_model(
    model, input_tensors=None, target_tensors=None, custom_objects=None,
    compile_clone=True, in_place_reset=False, optimizer_iterations=None,
    optimizer_config=None):
  """Clone a `Model` and build/compile it with the same settings used before.
  This function can be run in the same graph or in a separate graph from the
  model. When using a separate graph, `in_place_reset` must be `False`.
  Note that, currently, the clone produced from this function may not work with
  TPU DistributionStrategy. Try at your own risk.
  Args:
    model: `tf.keras.Model` object. Can be Functional, Sequential, or
      sub-classed.
    input_tensors: Optional list or dictionary of input tensors to build the
      model upon. If not provided, placeholders will be created.
    target_tensors: Optional list of target tensors for compiling the model. If
      not provided, placeholders will be created.
    custom_objects: Optional dictionary mapping string names to custom classes
      or functions.
    compile_clone: Boolean, whether to compile model clone (default `True`).
    in_place_reset: Boolean, whether to reset the model in place. Only used if
      the model is a subclassed model. In the case of a subclassed model,
      this argument must be set to `True` (default `False`). To restore the
      original model, use the function
      `in_place_subclassed_model_state_restoration(model)`.
    optimizer_iterations: An iterations variable that will be incremented by the
      optimizer if the clone is compiled. This argument is used when a Keras
      model is cloned into an Estimator model function, because Estimators
      create their own global step variable.
    optimizer_config: Optimizer config dictionary or list of dictionary
      returned from `get_config()`. This argument should be defined if
      `clone_and_build_model` is called in a different graph or session from
      the original model, and the optimizer is an instance of `OptimizerV2`.
  Returns:
    Clone of the model.
  Raises:
    ValueError: Cloning fails in the following cases
      - cloning a subclassed model with `in_place_reset` set to False.
      - compiling the clone when the original model has not been compiled.
  """
  # Grab optimizer now, as we reset-in-place for subclassed models, but
  # want to maintain access to the original optimizer.
  orig_optimizer = model.optimizer
  if compile_clone and not orig_optimizer:
    raise ValueError(
        'Error when cloning model: compile_clone was set to True, but the '
        'original model has not been compiled.')
  if compile_clone:
    compile_args = model._get_compile_args()  # pylint: disable=protected-access
    # Allows this method to be robust to switching graph and eager classes.
    model._get_compile_args = lambda: compile_args
  with CustomObjectScope(custom_objects or {}):
    if model._is_graph_network:
      clone = clone_model(model, input_tensors=input_tensors)
    elif isinstance(model, Sequential):
      clone = clone_model(model, input_tensors=input_tensors)
      if (not clone._is_graph_network and model._build_input_shape is not None):
        if ops.executing_eagerly_outside_functions():
          clone.build(model._build_input_shape)
        else:
          clone._set_inputs(
              backend.placeholder(
                  model._build_input_shape, dtype=model.inputs[0].dtype))
    else:
      try:
        # Prefer cloning the model if serial/deserial logic is implemented for
        # subclassed model.
        clone = model.__class__.from_config(model.get_config())
      except NotImplementedError:
        logging.warning('This model is a subclassed model. Please implement '
                        '`get_config` and `from_config` to better support '
                        'cloning the model.')
        if not in_place_reset:
          raise ValueError(
              'This model is a subclassed model. '
              'Such a model cannot be cloned, but there is a workaround where '
              'the model is reset in-place. To use this, please set the '
              'argument `in_place_reset` to `True`. This will reset the '
              'attributes in the original model. To restore the attributes, '
              'call `in_place_subclassed_model_state_restoration(model)`.')
        clone = model
        _in_place_subclassed_model_reset(clone)
      if input_tensors is not None:
        if isinstance(input_tensors, (list, tuple)) and len(input_tensors) == 1:
          input_tensors = input_tensors[0]
        clone._set_inputs(input_tensors)
  if compile_clone:
    if isinstance(orig_optimizer, optimizer_v1.TFOptimizer):
      optimizer = optimizer_v1.TFOptimizer(
          orig_optimizer.optimizer, optimizer_iterations)
      backend.track_tf_optimizer(optimizer)
    else:
      if not isinstance(orig_optimizer, (tuple, list)):
        orig_optimizer = [orig_optimizer]
      if optimizer_config is None:
        optimizer = [
            opt.__class__.from_config(opt.get_config())
            for opt in orig_optimizer
        ]
      elif isinstance(optimizer_config, dict):
        optimizer = [orig_optimizer[0].__class__.from_config(optimizer_config)]
      else:
        # optimizer config is list of dict, same order as orig_optimizer.
        optimizer = [
            opt.__class__.from_config(opt_config)
            for (opt, opt_config) in zip(orig_optimizer, optimizer_config)
        ]
      if optimizer_iterations is not None:
        for opt in optimizer:
          opt.iterations = optimizer_iterations
      if len(optimizer) == 1:
        optimizer = optimizer[0]
    compile_args['optimizer'] = optimizer
    if target_tensors is not None:
      compile_args['target_tensors'] = target_tensors
    # Ensure Metric objects in new model are separate from existing model.
    compile_args['metrics'] = metrics_module.clone_metrics(
        compile_args['metrics'])
    compile_args['weighted_metrics'] = metrics_module.clone_metrics(
        compile_args['weighted_metrics'])
    clone.compile(**compile_args)
  return clone
