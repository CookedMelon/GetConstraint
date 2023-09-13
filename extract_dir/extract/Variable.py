@tf_export("Variable", v1=[])
# TODO(mdan): This should subclass core.Tensor, and not all its subclasses?
class Variable(trackable.Trackable, metaclass=VariableMetaclass):
  """See the [variable guide](https://tensorflow.org/guide/variable).
  A variable maintains shared, persistent state manipulated by a program.
  The `Variable()` constructor requires an initial value for the variable, which
  can be a `Tensor` of any type and shape. This initial value defines the type
  and shape of the variable. After construction, the type and shape of the
  variable are fixed. The value can be changed using one of the assign methods.
  >>> v = tf.Variable(1.)
  >>> v.assign(2.)
  <tf.Variable ... shape=() dtype=float32, numpy=2.0>
  >>> v.assign_add(0.5)
  <tf.Variable ... shape=() dtype=float32, numpy=2.5>
  The `shape` argument to `Variable`'s constructor allows you to construct a
  variable with a less defined shape than its `initial_value`:
  >>> v = tf.Variable(1., shape=tf.TensorShape(None))
  >>> v.assign([[1.]])
  <tf.Variable ... shape=<unknown> dtype=float32, numpy=array([[1.]], ...)>
  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs to operations. Additionally, all the operators overloaded for the
  `Tensor` class are carried over to variables.
  >>> w = tf.Variable([[1.], [2.]])
  >>> x = tf.constant([[3., 4.]])
  >>> tf.matmul(w, x)
  <tf.Tensor:... shape=(2, 2), ... numpy=
    array([[3., 4.],
           [6., 8.]], dtype=float32)>
  >>> tf.sigmoid(w + x)
  <tf.Tensor:... shape=(2, 2), ...>
  When building a machine learning model it is often convenient to distinguish
  between variables holding trainable model parameters and other variables such
  as a `step` variable used to count training steps. To make this easier, the
  variable constructor supports a `trainable=<bool>`
  parameter. `tf.GradientTape` watches trainable variables by default:
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   trainable = tf.Variable(1.)
  ...   non_trainable = tf.Variable(2., trainable=False)
  ...   x1 = trainable * 2.
  ...   x2 = non_trainable * 3.
  >>> tape.gradient(x1, trainable)
  <tf.Tensor:... shape=(), dtype=float32, numpy=2.0>
  >>> assert tape.gradient(x2, non_trainable) is None  # Unwatched
  Variables are automatically tracked when assigned to attributes of types
  inheriting from `tf.Module`.
  >>> m = tf.Module()
  >>> m.v = tf.Variable([1.])
  >>> m.trainable_variables
  (<tf.Variable ... shape=(1,) ... numpy=array([1.], dtype=float32)>,)
  This tracking then allows saving variable values to
  [training checkpoints](https://www.tensorflow.org/guide/checkpoint), or to
  [SavedModels](https://www.tensorflow.org/guide/saved_model) which include
  serialized TensorFlow graphs.
  Variables are often captured and manipulated by `tf.function`s. This works the
  same way the un-decorated function would have:
  >>> v = tf.Variable(0.)
  >>> read_and_decrement = tf.function(lambda: v.assign_sub(0.1))
  >>> read_and_decrement()
  <tf.Tensor: shape=(), dtype=float32, numpy=-0.1>
  >>> read_and_decrement()
  <tf.Tensor: shape=(), dtype=float32, numpy=-0.2>
  Variables created inside a `tf.function` must be owned outside the function
  and be created only once:
  >>> class M(tf.Module):
  ...   @tf.function
  ...   def __call__(self, x):
  ...     if not hasattr(self, "v"):  # Or set self.v to None in __init__
  ...       self.v = tf.Variable(x)
  ...     return self.v * x
  >>> m = M()
  >>> m(2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=4.0>
  >>> m(3.)
  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  >>> m.v
  <tf.Variable ... shape=() dtype=float32, numpy=2.0>
  See the `tf.function` documentation for details.
  """
  @deprecated_args(
      None, "A variable's value can be manually cached by calling "
      "tf.Variable.read_value() under a tf.device scope. The caching_device "
      "argument does not work properly.", "caching_device")
  def __init__(self,
               initial_value=None,
               trainable=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               import_scope=None,
               constraint=None,
               synchronization=VariableSynchronization.AUTO,
               aggregation=VariableAggregation.NONE,
               shape=None,
               experimental_enable_variable_lifting=True,
               ):
    """Creates a new variable with value `initial_value`.
    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        variable. Defaults to `True`, unless `synchronization` is set to
        `ON_READ`, in which case it defaults to `False`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Note: This argument is only valid when using a v1-style
        `Session`. Optional device string describing where the Variable should
        be cached for reading. Defaults to the Variable's device. If not `None`,
        caches on another device. Typical use is to cache on the device where
        the Ops using the Variable reside, to deduplicate copying through
        `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      variable_def: `VariableDef` protocol buffer. If not `None`, recreates the
        Variable object with its contents, referencing the variable's nodes in
        the graph, which must already exist. The graph is not changed.
        `variable_def` and the other arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type. If
        `None`, either the datatype will be kept (if `initial_value` is a
        Tensor), or `convert_to_tensor` will decide.
      import_scope: Optional `string`. Name scope to add to the `Variable.` Only
        used when initializing from protocol buffer.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.
    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    raise NotImplementedError
  def __repr__(self):
    raise NotImplementedError
  def value(self):
    """Returns the last snapshot of this variable.
    You usually do not need to call this method as all ops that need the value
    of the variable call it automatically through a `convert_to_tensor()` call.
    Returns a `Tensor` which holds the value of the variable.  You can not
    assign a new value to this tensor as it is not a reference to the variable.
    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.
    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError
  def read_value(self):
    """Returns the value of this variable, read in the current context.
    Can be different from value() if it's on another device, with control
    dependencies, etc.
    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError
  def set_shape(self, shape):
    """Overrides the shape for this variable.
    Args:
      shape: the `TensorShape` representing the overridden shape.
    """
    raise NotImplementedError
  @property
  def trainable(self):
    raise NotImplementedError
  @property
  def synchronization(self):
    raise NotImplementedError
  @property
  def aggregation(self):
    raise NotImplementedError
  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.
    This is not a graph construction method, it does not add ops to the graph.
    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See `tf.compat.v1.Session` for more
    information on launching a graph and on sessions.
    ```python
    v = tf.Variable([1, 2])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```
    Args:
      session: The session to use to evaluate this variable. If none, the
        default session is used.
    Returns:
      A numpy `ndarray` with a copy of the value of this variable.
    """
    raise NotImplementedError
  @deprecated(
      None, "Use Variable.read_value. Variables in 2.X are initialized "
      "automatically both in eager and graph (inside tf.defun) contexts.")
  def initialized_value(self):
    """Returns the value of the initialized variable.
    You should use this instead of the variable itself to initialize another
    variable with a value that depends on the value of this variable.
    ```python
    # Initialize 'v' with a random tensor.
    v = tf.Variable(tf.random.truncated_normal([10, 40]))
    # Use `initialized_value` to guarantee that `v` has been
    # initialized before its value is used to initialize `w`.
    # The random values are picked only once.
    w = tf.Variable(v.initialized_value() * 2.0)
    ```
    Returns:
      A `Tensor` holding the value of this variable after its initializer
      has run.
    """
    raise NotImplementedError
  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable.
    Note that this is different from `initialized_value()` which runs
    the op that initializes the variable before returning its value.
    This method returns the tensor that is used by the op that initializes
    the variable.
    Returns:
      A `Tensor`.
    """
    raise NotImplementedError
  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.
    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    raise NotImplementedError
  def assign(self, value, use_locking=False, name=None, read_value=True):
    """Assigns a new value to the variable.
    This is essentially a shortcut for `assign(self, value)`.
    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.
    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError
  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    """Adds a value to this variable.
     This is essentially a shortcut for `assign_add(self, delta)`.
    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.
    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError
  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    """Subtracts a value from this variable.
    This is essentially a shortcut for `assign_sub(self, delta)`.
    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.
    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError
  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    """Subtracts `tf.IndexedSlices` from this variable.
    Args:
      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    """Adds `tf.IndexedSlices` to this variable.
    Args:
      sparse_delta: `tf.IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the max of `tf.IndexedSlices` and itself.
    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the min of `tf.IndexedSlices` and itself.
    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    """Multiply this variable by `tf.IndexedSlices`.
    Args:
      sparse_delta: `tf.IndexedSlices` to multiply this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    """Divide this variable by `tf.IndexedSlices`.
    Args:
      sparse_delta: `tf.IndexedSlices` to divide this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable.
    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable batch-wise.
    Analogous to `batch_gather`. This assumes that this variable and the
    sparse_delta IndexedSlices have a series of leading dimensions that are the
    same for all of them, and the updates are performed on the last dimension of
    indices. In other words, the dimensions should be the following:
    `num_prefix_dims = sparse_delta.indices.ndims - 1`
    `batch_dim = num_prefix_dims + 1`
    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[
         batch_dim:]`
    where
    `sparse_delta.updates.shape[:num_prefix_dims]`
    `== sparse_delta.indices.shape[:num_prefix_dims]`
    `== var.shape[:num_prefix_dims]`
    And the operation performed can be expressed as:
    `var[i_1, ..., i_n,
         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[
            i_1, ..., i_n, j]`
    When sparse_delta.indices is a 1D tensor, this operation is equivalent to
    `scatter_update`.
    To avoid this operation one can looping over the first `ndims` of the
    variable and using `scatter_update` on the subtensors that result of slicing
    the first dimension. This is a valid option for `ndims = 1`, but less
    efficient than this implementation.
    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError
  def scatter_nd_sub(self, indices, updates, name=None):
    """Applies sparse subtraction to individual values or slices in a Variable.
    Assuming the variable has rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_sub(indices, updates)
        print(v)
    ```
    After the update `v` would look like this:
        [1, -9, 3, -6, -4, 6, 7, -4]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    raise NotImplementedError
  def scatter_nd_add(self, indices, updates, name=None):
    """Applies sparse addition to individual values or slices in a Variable.
    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_add(indices, updates)
        print(v)
    ```
    The resulting update to v would look like this:
        [1, 13, 3, 14, 14, 6, 7, 20]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    raise NotImplementedError
  def scatter_nd_update(self, indices, updates, name=None):
    """Applies sparse assignment to individual values or slices in a Variable.
    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.
    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.
    `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```
    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:
    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_update(indices, updates)
        print(v)
    ```
    The resulting update to v would look like this:
        [1, 11, 3, 10, 9, 6, 7, 12]
    See `tf.scatter_nd` for more details about how to make updates to
    slices.
    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.
    Returns:
      The updated variable.
    """
    raise NotImplementedError
  def sparse_read(self, indices, name=None):
    r"""Gather slices from params axis axis according to indices.
    This function supports a subset of tf.gather, see tf.gather for details on
    usage.
    Args:
      indices: The index `Tensor`.  Must be one of the following types: `int32`,
        `int64`. Must be in range `[0, params.shape[axis])`.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `params`.
    """
    raise AttributeError
  def gather_nd(self, indices, name=None):
    r"""Gather slices from `params` into a Tensor with shape specified by `indices`.
    See tf.gather_nd for details.
    Args:
      indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
        Index tensor.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `params`.
    """
    raise AttributeError
  @deprecated(None, "Prefer Dataset.range instead.")
  def count_up_to(self, limit):
    """Increments this variable until it reaches `limit`.
    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.
    If no error is raised, the Op outputs the value of the variable before
    the increment.
    This is essentially a shortcut for `count_up_to(self, limit)`.
    Args:
      limit: value at which incrementing the variable raises an error.
    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
    raise NotImplementedError
  @deprecated(None,
              "Prefer Variable.assign which has equivalent behavior in 2.X.")
  def load(self, value, session=None):
    """Load new value into this variable.
    Writes new value to variable's memory. Doesn't add ops to the graph.
    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See `tf.compat.v1.Session` for more
    information on launching a graph and on sessions.
    ```python
    v = tf.Variable([1, 2])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        v.load([2, 3], sess)
        print(v.eval(sess)) # prints [2 3]
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        v.load([3, 4], sess)
        print(v.eval()) # prints [3 4]
    ```
    Args:
        value: New variable value
        session: The session to use to evaluate this variable. If none, the
          default session is used.
    Raises:
        ValueError: Session is not passed and no default session
    """
    if context.executing_eagerly():
      self.assign(value)
    else:
      session = session or ops.get_default_session()
      if session is None:
        raise ValueError(
            "Either session argument should be provided or default session "
            "should be established")
      session.run(self.initializer, {self.initializer.inputs[1]: value})
  # Conversion to tensor.
  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          f"Incompatible type conversion requested to type '{dtype.name}' for "
          f"variable of type '{v.dtype.name}' (Variable: {v}).")
    if as_ref:
      return v._ref()  # pylint: disable=protected-access
    else:
      return v.value()
  @classmethod
  def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      cls._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(cls, "__getitem__", array_ops._SliceHelperVar)
  @classmethod
  def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
    """Defer an operator overload to `ops.Tensor`.
    We pull the operator out of ops.Tensor dynamically to avoid ordering issues.
    Args:
      operator: string. The operator name.
    """
    # We can't use the overload mechanism on __eq__ & __ne__ since __eq__ is
    # called when adding a variable to sets. As a result we call a.value() which
    # causes infinite recursion when operating within a GradientTape
    # TODO(gjn): Consider removing this
    if operator == "__eq__" or operator == "__ne__":
      return
    tensor_oper = getattr(ops.Tensor, operator)
    def _run_op(a, *args, **kwargs):
      # pylint: disable=protected-access
      return tensor_oper(a.value(), *args, **kwargs)
    functools.update_wrapper(_run_op, tensor_oper)
    setattr(cls, operator, _run_op)
  def __hash__(self):
    if ops.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions():  # pylint: disable=protected-access
      raise TypeError(
          "Variable is unhashable. "
          f"Instead, use variable.ref() as the key. (Variable: {self})")
    else:
      return id(self)
  # TODO(gjn): duplicate of math_ops.tensor_equals, consider removing
  def __eq__(self, other):
    """Compares two variables element-wise for equality."""
    if ops.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions():  # pylint: disable=protected-access
      return gen_math_ops.equal(self, other, incompatible_shape_error=False)
    else:
      # In legacy graph mode, tensor equality is object equality
      return self is other
  # TODO(gjn): duplicate of math_ops.tensor_not_equals, consider removing
  def __ne__(self, other):
    """Compares two variables element-wise for equality."""
    if ops.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions():  # pylint: disable=protected-access
      return gen_math_ops.not_equal(self, other, incompatible_shape_error=False)
    else:
      # In legacy graph mode, tensor equality is object equality
      return self is not other
  def __iter__(self):
    """When executing eagerly, iterates over the value of the variable."""
    return iter(self.read_value())
  # NOTE(mrry): This enables the Variable's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Variable class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Variables interact
  # with ndarrays.
  __array_priority__ = 100
  @property
  def name(self):
    """The name of this variable."""
    raise NotImplementedError
  @property
  def _shared_name(self):
    """The shared name of the variable.
      Unlike name(), shared_name doesn't have ":0" suffix. It is user-specified
      name with name scope prefix.
    Returns:
      variable name.
    """
    return self.name[:self.name.index(":")]
  @property
  def initializer(self):
    """The initializer operation for this variable."""
    raise NotImplementedError
  @property
  def device(self):
    """The device of this variable."""
    raise NotImplementedError
  @property
  def dtype(self):
    """The `DType` of this variable."""
    raise NotImplementedError
  @property
  def op(self):
    """The `Operation` of this variable."""
    raise NotImplementedError
  @property
  def graph(self):
    """The `Graph` of this variable."""
    raise NotImplementedError
  @property
  def shape(self):
    """The `TensorShape` of this variable.
    Returns:
      A `TensorShape`.
    """
    raise NotImplementedError
  def get_shape(self):
    """Alias of `Variable.shape`."""
    return self.shape
  def _gather_saveables_for_checkpoint(self):
    """For implementing `Trackable`. This object is saveable on its own."""
    return {trackable.VARIABLE_VALUE_KEY: self}
  def to_proto(self, export_scope=None):
    """Converts a `Variable` to a `VariableDef` protocol buffer.
    Args:
      export_scope: Optional `string`. Name scope to remove.
    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    raise NotImplementedError
  @staticmethod
  def from_proto(variable_def, import_scope=None):
    """Returns a `Variable` object created from `variable_def`."""
    raise NotImplementedError
  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `Variable`.
    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info
  def _get_save_slice_info(self):
    return self._save_slice_info
  @deprecated(None, "Use ref() instead.")
  def experimental_ref(self):
    return self.ref()
  def ref(self):
    # tf.Tensor also has the same ref() API.  If you update the
    # documentation here, please update tf.Tensor.ref() as well.
    """Returns a hashable reference object to this Variable.
    The primary use case for this API is to put variables in a set/dictionary.
    We can't put variables in a set/dictionary as `variable.__hash__()` is no
    longer available starting Tensorflow 2.0.
    The following will raise an exception starting 2.0
    >>> x = tf.Variable(5)
    >>> y = tf.Variable(10)
    >>> z = tf.Variable(10)
    >>> variable_set = {x, y, z}
    Traceback (most recent call last):
      ...
    TypeError: Variable is unhashable. Instead, use tensor.ref() as the key.
    >>> variable_dict = {x: 'five', y: 'ten'}
    Traceback (most recent call last):
      ...
    TypeError: Variable is unhashable. Instead, use tensor.ref() as the key.
    Instead, we can use `variable.ref()`.
    >>> variable_set = {x.ref(), y.ref(), z.ref()}
    >>> x.ref() in variable_set
    True
    >>> variable_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
    >>> variable_dict[y.ref()]
    'ten'
    Also, the reference object provides `.deref()` function that returns the
    original Variable.
    >>> x = tf.Variable(5)
    >>> x.ref().deref()
    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=5>
    """
    return object_identity.Reference(self)
  @classmethod
  def _variable_call(
      cls,
      initial_value=None,
      trainable=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      import_scope=None,
      constraint=None,
      synchronization=VariableSynchronization.AUTO,
      aggregation=VariableAggregation.NONE,
      shape=None,
      experimental_enable_variable_lifting=None,
      **kwargs,
    ):
    """Variable class getter. Useful to force the signature."""
    if cls is not Variable:
      return None
    previous_getter = lambda **kws: default_variable_creator_v2(None, **kws)
    for _, getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
      previous_getter = _make_getter(getter, previous_getter)
    # Reset `aggregation` that is explicitly set as `None` to the enum NONE.
    if aggregation is None:
      aggregation = VariableAggregation.NONE
    return previous_getter(
        initial_value=initial_value,
        trainable=trainable,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        import_scope=import_scope,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
        experimental_enable_variable_lifting=experimental_enable_variable_lifting,
    )
  class SaveSliceInfo:
    """Information on how to save this Variable as a slice.
    Provides internal support for saving variables as slices of a larger
    variable.  This API is not public and is subject to change.
    Available properties:
    * full_name
    * full_shape
    * var_offset
    * var_shape
    """
    def __init__(self,
                 full_name=None,
                 full_shape=None,
                 var_offset=None,
                 var_shape=None,
                 save_slice_info_def=None,
                 import_scope=None):
      """Create a `SaveSliceInfo`.
      Args:
        full_name: Name of the full variable of which this `Variable` is a
          slice.
        full_shape: Shape of the full variable, as a list of int.
        var_offset: Offset of this `Variable` into the full variable, as a list
          of int.
        var_shape: Shape of this `Variable`, as a list of int.
        save_slice_info_def: `SaveSliceInfoDef` protocol buffer. If not `None`,
          recreates the SaveSliceInfo object its contents. `save_slice_info_def`
          and other arguments are mutually exclusive.
        import_scope: Optional `string`. Name scope to add. Only used when
          initializing from protocol buffer.
      """
      if save_slice_info_def:
        assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
        self.full_name = ops.prepend_name_scope(
            save_slice_info_def.full_name, import_scope=import_scope)
        self.full_shape = list(save_slice_info_def.full_shape)
        self.var_offset = list(save_slice_info_def.var_offset)
        self.var_shape = list(save_slice_info_def.var_shape)
      else:
        self.full_name = full_name
        self.full_shape = full_shape
        self.var_offset = var_offset
        self.var_shape = var_shape
    @property
    def spec(self):
      """Computes the spec string used for saving."""
      full_shape_str = " ".join("%d" % d for d in self.full_shape) + " "
      sl_spec = ":".join(
          "%d,%d" % (o, s) for o, s in zip(self.var_offset, self.var_shape))
      return full_shape_str + sl_spec
    def to_proto(self, export_scope=None):
      """Returns a SaveSliceInfoDef() proto.
      Args:
        export_scope: Optional `string`. Name scope to remove.
      Returns:
        A `SaveSliceInfoDef` protocol buffer, or None if the `Variable` is not
        in the specified name scope.
      """
      if (export_scope is None or self.full_name.startswith(export_scope)):
        save_slice_info_def = variable_pb2.SaveSliceInfoDef()
        save_slice_info_def.full_name = ops.strip_name_scope(
            self.full_name, export_scope)
        for i in self.full_shape:
          save_slice_info_def.full_shape.append(i)
        for i in self.var_offset:
          save_slice_info_def.var_offset.append(i)
        for i in self.var_shape:
          save_slice_info_def.var_shape.append(i)
        return save_slice_info_def
      else:
        return None
