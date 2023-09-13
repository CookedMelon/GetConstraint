@tf_export("Module")
class Module(autotrackable.AutoTrackable):
  """Base neural network module class.
  A module is a named container for `tf.Variable`s, other `tf.Module`s and
  functions which apply to user input. For example a dense layer in a neural
  network might be implemented as a `tf.Module`:
  >>> class Dense(tf.Module):
  ...   def __init__(self, input_dim, output_size, name=None):
  ...     super().__init__(name=name)
  ...     self.w = tf.Variable(
  ...       tf.random.normal([input_dim, output_size]), name='w')
  ...     self.b = tf.Variable(tf.zeros([output_size]), name='b')
  ...   def __call__(self, x):
  ...     y = tf.matmul(x, self.w) + self.b
  ...     return tf.nn.relu(y)
  You can use the Dense layer as you would expect:
  >>> d = Dense(input_dim=3, output_size=2)
  >>> d(tf.ones([1, 3]))
  <tf.Tensor: shape=(1, 2), dtype=float32, numpy=..., dtype=float32)>
  By subclassing `tf.Module` instead of `object` any `tf.Variable` or
  `tf.Module` instances assigned to object properties can be collected using
  the `variables`, `trainable_variables` or `submodules` property:
  >>> d.variables
      (<tf.Variable 'b:0' shape=(2,) dtype=float32, numpy=...,
      dtype=float32)>,
      <tf.Variable 'w:0' shape=(3, 2) dtype=float32, numpy=..., dtype=float32)>)
  Subclasses of `tf.Module` can also take advantage of the `_flatten` method
  which can be used to implement tracking of any other types.
  All `tf.Module` classes have an associated `tf.name_scope` which can be used
  to group operations in TensorBoard and create hierarchies for variable names
  which can help with debugging. We suggest using the name scope when creating
  nested submodules/parameters or for forward methods whose graph you might want
  to inspect in TensorBoard. You can enter the name scope explicitly using
  `with self.name_scope:` or you can annotate methods (apart from `__init__`)
  with `@tf.Module.with_name_scope`.
  >>> class MLP(tf.Module):
  ...   def __init__(self, input_size, sizes, name=None):
  ...     super().__init__(name=name)
  ...     self.layers = []
  ...     with self.name_scope:
  ...       for size in sizes:
  ...         self.layers.append(Dense(input_dim=input_size, output_size=size))
  ...         input_size = size
  ...   @tf.Module.with_name_scope
  ...   def __call__(self, x):
  ...     for layer in self.layers:
  ...       x = layer(x)
  ...     return x
  >>> module = MLP(input_size=5, sizes=[5, 5])
  >>> module.variables
  (<tf.Variable 'mlp/b:0' shape=(5,) dtype=float32, numpy=..., dtype=float32)>,
  <tf.Variable 'mlp/w:0' shape=(5, 5) dtype=float32, numpy=...,
     dtype=float32)>,
  <tf.Variable 'mlp/b:0' shape=(5,) dtype=float32, numpy=..., dtype=float32)>,
  <tf.Variable 'mlp/w:0' shape=(5, 5) dtype=float32, numpy=...,
     dtype=float32)>)
  """
  # AutoTrackable adds object attributes that users will not expect us to
  # include when flattening (these reference dependencies reachable via other
  # object attributes).
  _TF_MODULE_IGNORED_PROPERTIES = frozenset((
      "_self_unconditional_checkpoint_dependencies",
      "_self_unconditional_dependency_names"
  ))
  def __init__(self, name=None):
    if name is None:
      name = camel_to_snake(type(self).__name__)
    else:
      if not valid_identifier(name):
        raise ValueError(
            "%r is not a valid module name. Module names must be valid Python "
            "identifiers (e.g. a valid class name)." % name)
    self._name = name
    if tf2.enabled():
      with ops.name_scope_v2(name) as scope_name:
        self._name_scope = ops.name_scope_v2(scope_name)
    else:
      with ops.name_scope(name, skip_on_eager=False) as scope_name:
        self._scope_name = scope_name
  @property
  def name(self):
    """Returns the name of this module as passed or determined in the ctor.
    NOTE: This is not the same as the `self.name_scope.name` which includes
    parent module names.
    """
    return self._name
  @property
  def name_scope(self):
    """Returns a `tf.name_scope` instance for this class."""
    if tf2.enabled():
      return self._name_scope
    else:
      # In TF1 name_scope is not re-entrant in eager so we cannot memoize it.
      return ops.name_scope(self._scope_name, skip_on_eager=False)
  @property
  def variables(self):
    """Sequence of variables owned by this module and its submodules.
    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.
    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """
    return tuple(self._flatten(predicate=_is_variable, expand_composites=True))
  @property
  def trainable_variables(self):
    """Sequence of trainable variables owned by this module and its submodules.
    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.
    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """
    return tuple(
        self._flatten(predicate=_is_trainable_variable, expand_composites=True))
  @property
  def non_trainable_variables(self):
    """Sequence of non-trainable variables owned by this module and its submodules.
    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.
    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """
    return tuple(self._flatten(
        predicate=_is_non_trainable_variable, expand_composites=True))
  @property
  def submodules(self):
    """Sequence of all sub-modules.
    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).
    >>> a = tf.Module()
    >>> b = tf.Module()
    >>> c = tf.Module()
    >>> a.b = b
    >>> b.c = c
    >>> list(a.submodules) == [b, c]
    True
    >>> list(b.submodules) == [c]
    True
    >>> list(c.submodules) == []
    True
    Returns:
      A sequence of all submodules.
    """
    return tuple(self._flatten(predicate=_is_module))
  def _flatten(self,
               recursive=True,
               predicate=None,
               attribute_traversal_key=None,
               with_path=False,
               expand_composites=False):
    """Flattened attribute values in sorted order by attribute name.
    Modules are flattened by first walking their attributes in name order.
    Each attribute value is then flattened to find leaf values. If flatten is
    applied `recursive`ly and if the leaf is a `Module` it will also be
    flattened to find leaves. Finally every leaf value is optionally tested
    against the given `predicate` and finally yielded.
    ```
    class Foo(tf.Module):
      def __init__(self):
        super().__init__()
        self.x = [tf.constant('a'), tf.constant('b')]
        self.y = {'i': tf.constant('c'), 'j': tf.constant('d')}
        self.z = tf.constant('e')
      @property
      def tensors(self):
        return tuple(self._flatten(predicate=is_tensor, with_path=True))
    foo = Foo()
    foo.tensors
    # ==> ((('x', 0),   <tf.Tensor: ...'a'>),
    #     (('x', 1),   <tf.Tensor: ...'b'>),
    #     (('y', 'i'), <tf.Tensor: ...'c'>),
    #     (('y', 'j'), <tf.Tensor: ...'d'>),
    #     (('z',),     <tf.Tensor: ...'e'>))
    ```
    `attribute_traversal_key` controls the order object properties are visited.
    If not set objects are visited in ascending order by name.
    Args:
      recursive: Whether to recurse into child modules or not.
      predicate: (Optional) If set then only values matching predicate are
        yielded. A value of `None` (the default) means no items will be
        filtered.
      attribute_traversal_key: (Optional) Method to rekey object attributes
        before they are sorted. Contract is the same as `key` argument to
        builtin `sorted` and only applies to object properties.
      with_path: (Optional) Whether to include the path to the object as well
        as the object itself. If `with_path` is `True` then leaves will not be
        de-duplicated (e.g. if the same leaf instance is reachable via multiple
        modules then it will be yielded multiple times with different paths).
      expand_composites: If true, then composite tensors are expanded into their
        component tensors.
    Returns:
      Flat generator for leaves of the current module and optionally all
      submodules.
    """
    if predicate is None:
      predicate = lambda _: True
    return _flatten_module(
        self,
        recursive=recursive,
        predicate=predicate,
        attributes_to_ignore=self._TF_MODULE_IGNORED_PROPERTIES,
        attribute_traversal_key=attribute_traversal_key,
        with_path=with_path,
        expand_composites=expand_composites)
  @classmethod
  def with_name_scope(cls, method):
    """Decorator to automatically enter the module name scope.
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)
    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:
    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>
    Args:
      method: The method to wrap.
    Returns:
      The original method wrapped such that it enters the module's name scope.
    """
    def method_with_name_scope(self, *args, **kwargs):
      with self.name_scope:
        return method(self, *args, **kwargs)
    return tf_decorator.make_decorator(method, method_with_name_scope)
