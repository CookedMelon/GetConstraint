@tf_export("saved_model.SaveOptions")
class SaveOptions:
  """Options for saving to SavedModel.
  This function may be used in the `options` argument in functions that
  save a SavedModel (`tf.saved_model.save`, `tf.keras.models.save_model`).
  """
  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = ("namespace_whitelist", "save_debug_info", "function_aliases",
               "experimental_io_device", "experimental_variable_policy",
               "experimental_custom_gradients")
  def __init__(self,
               namespace_whitelist=None,
               save_debug_info=False,
               function_aliases=None,
               experimental_io_device=None,
               experimental_variable_policy=None,
               experimental_custom_gradients=True):
    """Creates an object that stores options for SavedModel saving.
    Args:
      namespace_whitelist: List of strings containing op namespaces to whitelist
        when saving a model. Saving an object that uses namespaced ops must
        explicitly add all namespaces to the whitelist. The namespaced ops must
        be registered into the framework when loading the SavedModel. If no
        whitelist is provided, all namespaced ops will be allowed.
      save_debug_info: Boolean indicating whether debug information is saved. If
        True, then a debug/saved_model_debug_info.pb file will be written with
        the contents of a GraphDebugInfo binary protocol buffer containing stack
        trace information for all ops and functions that are saved.
      function_aliases: Python dict. Mapping from string to object returned by
        @tf.function. A single tf.function can generate many ConcreteFunctions.
        If a downstream tool wants to refer to all concrete functions generated
        by a single tf.function you can use the `function_aliases` argument to
        store a map from the alias name to all concrete function names.
        E.g.
        >>> class Adder(tf.Module):
        ...   @tf.function
        ...   def double(self, x):
        ...     return x + x
        >>> model = Adder()
        >>> model.double.get_concrete_function(
        ...   tf.TensorSpec(shape=[], dtype=tf.float32, name="float_input"))
        >>> model.double.get_concrete_function(
        ...   tf.TensorSpec(shape=[], dtype=tf.string, name="string_input"))
        >>> options = tf.saved_model.SaveOptions(
        ...   function_aliases={'double': model.double})
        >>> tf.saved_model.save(model, '/tmp/adder', options=options)
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.
        This is for example useful if you want to save to a local directory,
        such as "/tmp" when running in a distributed setting. In that case pass
        a device for the host where the "/tmp" directory is accessible.
      experimental_variable_policy: The policy to apply to variables when
        saving. This is either a `saved_model.experimental.VariablePolicy` enum
        instance or one of its value strings (case is not important). See that
        enum documentation for details. A value of `None` corresponds to the
        default policy.
      experimental_custom_gradients: Boolean. When True, will save traced
        gradient functions for the functions decorated by `tf.custom_gradient`.
        Defaults to `True`.
    """
    self.namespace_whitelist = _validate_namespace_whitelist(
        namespace_whitelist)
    self.save_debug_info = save_debug_info
    self.function_aliases = function_aliases if function_aliases else dict()
    self.experimental_custom_gradients = experimental_custom_gradients
    self.experimental_io_device = experimental_io_device
    self.experimental_variable_policy = (
        VariablePolicy.from_obj(experimental_variable_policy))
