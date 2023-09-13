"/home/cc/Workspace/tfconstraint/python/saved_model/save.py"
@tf_export(
    "saved_model.save",
    v1=["saved_model.save", "saved_model.experimental.save"])
def save(obj, export_dir, signatures=None, options=None):
  # pylint: disable=line-too-long
  """Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) `obj` to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).
  The `obj` must inherit from the [`Trackable`
  class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/base.py#L591).
  Example usage:
  >>> class Adder(tf.Module):
  ...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
  ...   def add(self, x):
  ...     return x + x
  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')
  The resulting SavedModel is then servable with an input named "x", a scalar
  with dtype float32.
  _Signatures_
  Signatures define the input and output types for a computation. The optional
  save `signatures` argument controls which methods in `obj` will be
  available to programs which consume `SavedModel`s, for example, serving
  APIs. Python functions may be decorated with
  `@tf.function(input_signature=...)` and passed as signatures directly, or
  lazily with a call to `get_concrete_function` on the method decorated with
  `@tf.function`.
  Example:
  >>> class Adder(tf.Module):
  ...   @tf.function
  ...   def add(self, x):
  ...     return x + x
  >>> model = Adder()
  >>> tf.saved_model.save(
  ...   model, '/tmp/adder',signatures=model.add.get_concrete_function(
  ...     tf.TensorSpec([], tf.float32)))
  If a `@tf.function` does not have an input signature and
  `get_concrete_function` is not called on that method, the function will not
  be directly callable in the restored SavedModel.
  Example:
  >>> class Adder(tf.Module):
  ...   @tf.function
  ...   def add(self, x):
  ...     return x + x
  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')
  >>> restored = tf.saved_model.load('/tmp/adder')
  >>> restored.add(1.)
  Traceback (most recent call last):
  ...
  ValueError: Found zero restored functions for caller function.
  If the `signatures` argument is omitted, `obj` will be searched for
  `@tf.function`-decorated methods. If exactly one traced `@tf.function` is
  found, that method will be used as the default signature for the SavedModel.
  Else, any `@tf.function` attached to `obj` or its dependencies will be
  exported for use with `tf.saved_model.load`.
  When invoking a signature in an exported SavedModel, `Tensor` arguments are
  identified by name. These names will come from the Python function's argument
  names by default. They may be overridden by specifying a `name=...` argument
  in the corresponding `tf.TensorSpec` object. Explicit naming is required if
  multiple `Tensor`s are passed through a single argument to the Python
  function.
  The outputs of functions used as `signatures` must either be flat lists, in
  which case outputs will be numbered, or a dictionary mapping string keys to
  `Tensor`, in which case the keys will be used to name outputs.
  Signatures are available in objects returned by `tf.saved_model.load` as a
  `.signatures` attribute. This is a reserved attribute: `tf.saved_model.save`
  on an object with a custom `.signatures` attribute will raise an exception.
  _Using `tf.saved_model.save` with Keras models_
  While Keras has its own [saving and loading
  API](https://www.tensorflow.org/guide/keras/save_and_serialize),
  this function can be used to export Keras models. For example, exporting with
  a signature specified:
  >>> class Adder(tf.keras.Model):
  ...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  ...   def concat(self, x):
  ...      return x + x
  >>> model = Adder()
  >>> tf.saved_model.save(model, '/tmp/adder')
  Exporting from a function without a fixed signature:
  >>> class Adder(tf.keras.Model):
  ...   @tf.function
  ...   def concat(self, x):
  ...      return x + x
  >>> model = Adder()
  >>> tf.saved_model.save(
  ...   model, '/tmp/adder',
  ...   signatures=model.concat.get_concrete_function(
  ...     tf.TensorSpec(shape=[], dtype=tf.string, name="string_input")))
  `tf.keras.Model` instances constructed from inputs and outputs already have a
  signature and so do not require a `@tf.function` decorator or a `signatures`
  argument. If neither are specified, the model's forward pass is exported.
  >>> x = tf.keras.layers.Input((4,), name="x")
  >>> y = tf.keras.layers.Dense(5, name="out")(x)
  >>> model = tf.keras.Model(x, y)
  >>> tf.saved_model.save(model, '/tmp/saved_model/')
  The exported SavedModel takes "x" with shape [None, 4] and returns "out"
  with shape [None, 5]
  _Variables and Checkpoints_
  Variables must be tracked by assigning them to an attribute of a tracked
  object or to an attribute of `obj` directly. TensorFlow objects (e.g. layers
  from `tf.keras.layers`, optimizers from `tf.train`) track their variables
  automatically. This is the same tracking scheme that `tf.train.Checkpoint`
  uses, and an exported `Checkpoint` object may be restored as a training
  checkpoint by pointing `tf.train.Checkpoint.restore` to the SavedModel's
  "variables/" subdirectory.
  `tf.function` does not hard-code device annotations from outside the function
  body, instead of using the calling context's device. This means for example
  that exporting a model that runs on a GPU and serving it on a CPU will
  generally work, with some exceptions:
    * `tf.device` annotations inside the body of the function will be hard-coded
      in the exported model; this type of annotation is discouraged.
    * Device-specific operations, e.g. with "cuDNN" in the name or with
      device-specific layouts, may cause issues.
    * For `ConcreteFunctions`, active distribution strategies will cause device
      placements to be hard-coded in the function.
  SavedModels exported with `tf.saved_model.save` [strip default-valued
  attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes)
  automatically, which removes one source of incompatibilities when the consumer
  of a SavedModel is running an older TensorFlow version than the
  producer. There are however other sources of incompatibilities which are not
  handled automatically, such as when the exported model contains operations
  which the consumer does not have definitions for.
  Args:
    obj: A trackable object (e.g. tf.Module or tf.train.Checkpoint) to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: Optional, one of three types:
      * A `tf.function` with an input signature specified, which will use the
        default serving signature key.
      * The result of `f.get_concrete_function` on a `@tf.function`-decorated
        function `f`, in which case `f` will be used to generate a signature for
        the SavedModel under the default serving signature key.
      * A dictionary, which maps signature keys to either `tf.function`
        instances with input signatures or concrete functions. Keys of such a
        dictionary may be arbitrary strings, but will typically be from the
        `tf.saved_model.signature_constants` module.
    options: `tf.saved_model.SaveOptions` object for configuring save options.
  Raises:
    ValueError: If `obj` is not trackable.
  @compatibility(eager)
  Not well supported when graph building. From TensorFlow 1.x,
  `tf.compat.v1.enable_eager_execution()` should run first. Calling
  tf.saved_model.save in a loop when graph building from TensorFlow 1.x will
  add new save operations to the default graph each iteration.
  May not be called from within a function body.
  @end_compatibility
  """
  if isinstance(export_dir, os.PathLike):
    export_dir = os.fspath(export_dir)
  # pylint: enable=line-too-long
  metrics.IncrementWriteApi(_SAVE_V2_LABEL)
  save_and_return_nodes(obj, export_dir, signatures, options)
  metrics.IncrementWrite(write_version="2")
def save_and_return_nodes(obj,
                          export_dir,
                          signatures=None,
                          options=None,
                          experimental_skip_checkpoint=False):
  """Saves a SavedModel while returning all saved nodes and their paths.
  Please see `tf.saved_model.save` for details.
  Args:
    obj: A trackable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: A function or dictionary of functions to save in the SavedModel
      as signatures.
    options: `tf.saved_model.SaveOptions` object for configuring save options.
    experimental_skip_checkpoint: If set to `True`, the checkpoint will not be
      written.
  Returns:
    A tuple of (a list of saved nodes in the order they are serialized to the
      `SavedObjectGraph`, dictionary mapping nodes to one possible path from
      the root node to the key node)
  """
  options = options or save_options.SaveOptions()
  saved_model = saved_model_pb2.SavedModel()
  meta_graph_def = saved_model.meta_graphs.add()
  _, exported_graph, object_saver, asset_info, saved_nodes, node_paths = (
      _build_meta_graph(obj, signatures, options, meta_graph_def))
  saved_model.saved_model_schema_version = (
      constants.SAVED_MODEL_SCHEMA_VERSION)
  # Write the checkpoint, copy assets into the assets directory, and write out
  # the SavedModel proto itself.
  if not experimental_skip_checkpoint:
    path_helpers.get_or_create_variables_dir(export_dir)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_io_device=options.experimental_io_device)
    object_saver.save(
        path_helpers.get_variables_path(export_dir), options=ckpt_options)
  builder_impl.copy_assets_to_destination_dir(asset_info.asset_filename_map,
                                              export_dir)
  # Note that this needs to be the last file operation when saving the
  # SavedModel. Users rely on checking saved_model_dir/saved_model.pb as an
  # indication that the SavedModel is completely written.
  if context.executing_eagerly():
    try:
      context.async_wait()  # Ensure save operations have completed.
    except errors.NotFoundError as err:
      raise FileNotFoundError(
          f"{err}\n You may be trying to save on a different device from the "
          "computational device. Consider setting the "
          "`experimental_io_device` option in `tf.saved_model.SaveOptions` "
          "to the io_device such as '/job:localhost'.") from err
  # We will slowly migrate code in this function to pywrap_saved_model.Save
  # as we build up the C++ API.
  pywrap_saved_model.Save(export_dir)
  saved_model_serialized = saved_model.SerializeToString(deterministic=True)
  fingerprinting_utils.write_fingerprint(export_dir, saved_model_serialized)
  path = file_io.join(
      compat.as_str(export_dir),
      compat.as_str(constants.SAVED_MODEL_FILENAME_PB))
  file_io.atomic_write_string_to_file(path, saved_model_serialized)
  # Save debug info, if requested.
  if options.save_debug_info:
    _export_debug_info(exported_graph, export_dir)
  # For privacy concerns, please see the note in
  #  tensorflow/cc/saved_model/metrics.h
  metrics.SetWritePath(saved_model_path=str(export_dir))
  # Clean reference cycles so repeated export()s don't make work for the garbage
  # collector. Before this point, we need to keep references to captured
  # constants in the saved graph.
  ops.dismantle_graph(exported_graph)
  return saved_nodes, node_paths
def export_meta_graph(obj, filename, signatures=None, options=None):
  """Exports the MetaGraph proto of the `obj` to a file.
  This function goes through the same procedures saved_model.save goes to
  produce the given object's MetaGraph, then saves it to the given file. It
  skips saving checkpoint information, and is useful when all one wants is the
  graph defining the model.
  Args:
    obj: A trackable object to build the MetaGraph from.
    filename: The file into which to write the MetaGraph.
    signatures: Optional, either a `tf.function` with an input signature
      specified or the result of `f.get_concrete_function` on a
      `@tf.function`-decorated function `f`, in which case `f` will be used to
      generate a signature for the SavedModel under the default serving
      signature key. `signatures` may also be a dictionary, in which case it
      maps from signature keys to either `tf.function` instances with input
      signatures or concrete functions. The keys of such a dictionary may be
      arbitrary strings, but will typically be from the
      `tf.saved_model.signature_constants` module.
    options: Optional, `tf.saved_model.SaveOptions` object that specifies
      options for saving.
  """
  options = options or save_options.SaveOptions()
  export_dir = os.path.dirname(filename)
  meta_graph_def, exported_graph, _, _, _, _ = _build_meta_graph(
      obj, signatures, options)
  file_io.atomic_write_string_to_file(
      filename, meta_graph_def.SerializeToString(deterministic=True))
  # Save debug info, if requested.
  if options.save_debug_info:
    _export_debug_info(exported_graph, export_dir)
  # Clean reference cycles so repeated export()s don't make work for the garbage
  # collector. Before this point, we need to keep references to captured
  # constants in the saved graph.
  ops.dismantle_graph(exported_graph)
def _build_meta_graph_impl(obj, signatures, options, meta_graph_def=None):
  """Creates a MetaGraph containing the resources and functions of an object."""
  if ops.inside_function():
    raise AssertionError(
        "`tf.saved_model.save` is not supported inside a traced @tf.function. "
        "Move the call to the outer eagerly-executed context.")
  # pylint: enable=line-too-long
  if not isinstance(obj, base.Trackable):
    raise ValueError(
        "Expected an object of type `Trackable`, such as `tf.Module` or a "
        f"subclass of the `Trackable` class, for export. Got {obj} "
        f"with type {type(obj)}.")
  meta_graph_def = meta_graph_def or meta_graph_pb2.MetaGraphDef()
  augmented_graph_view = _AugmentedGraphView(obj)
  if signatures is None:
    signatures = signature_serialization.find_function_to_export(
        augmented_graph_view)
  signatures, wrapped_functions, defaults = (
      signature_serialization.canonicalize_signatures(signatures)
  )
  signature_serialization.validate_augmented_graph_view(augmented_graph_view)
  signature_map = signature_serialization.create_signature_map(signatures)
  augmented_graph_view.set_signature(signature_map, wrapped_functions)
  # Use _SaveableView to provide a frozen listing of properties and functions.
  saveable_view = _SaveableView(augmented_graph_view, options)
  object_saver = checkpoint.TrackableSaver(augmented_graph_view)
  asset_info, exported_graph = _fill_meta_graph_def(
      meta_graph_def,
      saveable_view,
      signatures,
      options.namespace_whitelist,
      options.experimental_custom_gradients,
      defaults,
  )
  if options.function_aliases:
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    for alias, func in options.function_aliases.items():
      if isinstance(func, types_core.ConcreteFunction):
        function_aliases[func.name] = alias
      elif isinstance(func, polymorphic_function.Function):
        for fdef in func._list_all_concrete_functions():  # pylint: disable=protected-access
          function_aliases[fdef.name] = alias
      else:
        raise TypeError(
            f"Unsupported type f{type(func)}. Functions in `function_aliases`"
            " should be created by tf.function, or concrete functions."
        )
  object_graph_proto = _serialize_object_graph(saveable_view,
                                               asset_info.asset_index)
  meta_graph_def.object_graph_def.CopyFrom(object_graph_proto)
  return (meta_graph_def, exported_graph, object_saver, asset_info,
          saveable_view.nodes, saveable_view.node_paths)
def _build_meta_graph(obj, signatures, options, meta_graph_def=None):
  """Creates a MetaGraph under a save context.
  Args:
    obj: A trackable object to build the MetaGraph from.
    signatures: Can be a `tf.function` with an input signature specified or the
      result of `f.get_concrete_function` on a `@tf.function`-decorated function
      `f`. `signatures` may also be a dictionary, in which case it maps from
      signature keys to `tf.function` instances. If None, finds signature to
      export from the `@tf.function`-decorated methods in `obj`.
    options: `tf.saved_model.SaveOptions` object that specifies options for
      saving.
    meta_graph_def: Optional, the MetaGraphDef proto fill.
  Raises:
    AssertionError: If `export_meta_graph` is executing inside a `tf.function`.
    ValueError: If `obj` is not trackable.
  Returns:
    meta_graph_def: Filled MetaGraphDef proto
    exported_graph: `tf.Graph` object generated from `obj`.
    object_saver: `checkpoint.TrackableSaver` of the `obj` and its dependencies.
    asset_info: `_AssetInfo` tuple containing external assets in the `obj`.
    saveable_view.nodes: _SaveableView nodes.
    saveable_view.node_paths: _SaveableView paths.
  """
  with save_context.save_context(options):
    return _build_meta_graph_impl(obj, signatures, options, meta_graph_def)
