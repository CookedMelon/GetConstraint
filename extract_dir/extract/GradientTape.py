@tf_export("GradientTape", "autodiff.GradientTape", v1=["GradientTape"])
class GradientTape:
  """Record operations for automatic differentiation.
  Operations are recorded if they are executed within this context manager and
  at least one of their inputs is being "watched".
  Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
  where `trainable=True` is default in both cases) are automatically watched.
  Tensors can be manually watched by invoking the `watch` method on this context
  manager.
  For example, consider the function `y = x * x`. The gradient at `x = 3.0` can
  be computed as:
  >>> x = tf.constant(3.0)
  >>> with tf.GradientTape() as g:
  ...   g.watch(x)
  ...   y = x * x
  >>> dy_dx = g.gradient(y, x)
  >>> print(dy_dx)
  tf.Tensor(6.0, shape=(), dtype=float32)
  GradientTapes can be nested to compute higher-order derivatives. For example,
  >>> x = tf.constant(5.0)
  >>> with tf.GradientTape() as g:
  ...   g.watch(x)
  ...   with tf.GradientTape() as gg:
  ...     gg.watch(x)
  ...     y = x * x
  ...   dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
  >>> d2y_dx2 = g.gradient(dy_dx, x)  # d2y_dx2 = 2
  >>> print(dy_dx)
  tf.Tensor(10.0, shape=(), dtype=float32)
  >>> print(d2y_dx2)
  tf.Tensor(2.0, shape=(), dtype=float32)
  By default, the resources held by a GradientTape are released as soon as
  GradientTape.gradient() method is called. To compute multiple gradients over
  the same computation, create a persistent gradient tape. This allows multiple
  calls to the gradient() method as resources are released when the tape object
  is garbage collected. For example:
  >>> x = tf.constant(3.0)
  >>> with tf.GradientTape(persistent=True) as g:
  ...   g.watch(x)
  ...   y = x * x
  ...   z = y * y
  >>> dz_dx = g.gradient(z, x)  # (4*x^3 at x = 3)
  >>> print(dz_dx)
  tf.Tensor(108.0, shape=(), dtype=float32)
  >>> dy_dx = g.gradient(y, x)
  >>> print(dy_dx)
  tf.Tensor(6.0, shape=(), dtype=float32)
  By default GradientTape will automatically watch any trainable variables that
  are accessed inside the context. If you want fine grained control over which
  variables are watched you can disable automatic tracking by passing
  `watch_accessed_variables=False` to the tape constructor:
  >>> x = tf.Variable(2.0)
  >>> w = tf.Variable(5.0)
  >>> with tf.GradientTape(
  ...     watch_accessed_variables=False, persistent=True) as tape:
  ...   tape.watch(x)
  ...   y = x ** 2  # Gradients will be available for `x`.
  ...   z = w ** 3  # No gradients will be available as `w` isn't being watched.
  >>> dy_dx = tape.gradient(y, x)
  >>> print(dy_dx)
  tf.Tensor(4.0, shape=(), dtype=float32)
  >>> # No gradients will be available as `w` isn't being watched.
  >>> dz_dw = tape.gradient(z, w)
  >>> print(dz_dw)
  None
  Note that when using models you should ensure that your variables exist when
  using `watch_accessed_variables=False`. Otherwise it's quite easy to make your
  first iteration not have any gradients:
  ```python
  a = tf.keras.layers.Dense(32)
  b = tf.keras.layers.Dense(32)
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a.variables)  # Since `a.build` has not been called at this point
                             # `a.variables` will return an empty list and the
                             # tape will not be watching anything.
    result = b(a(inputs))
    tape.gradient(result, a.variables)  # The result of this computation will be
                                        # a list of `None`s since a's variables
                                        # are not being watched.
  ```
  Note that only tensors with real or complex dtypes are differentiable.
  """
  def __init__(self, persistent=False, watch_accessed_variables=True):
    """Creates a new GradientTape.
    Args:
      persistent: Boolean controlling whether a persistent gradient tape
        is created. False by default, which means at most one call can
        be made to the gradient() method on this object.
      watch_accessed_variables: Boolean controlling whether the tape will
        automatically `watch` any (trainable) variables accessed while the tape
        is active. Defaults to True meaning gradients can be requested from any
        result computed in the tape derived from reading a trainable `Variable`.
        If False users must explicitly `watch` any `Variable`s they want to
        request gradients from.
    """
    self._tape = None
    self._persistent = persistent
    self._watch_accessed_variables = watch_accessed_variables
    self._watched_variables = ()
    self._recording = False
  def __enter__(self):
    """Enters a context inside which operations are recorded on this tape."""
    self._push_tape()
    return self
  def __exit__(self, typ, value, traceback):
    """Exits the recording context, no further operations are traced."""
    if self._recording:
      self._pop_tape()
  def _push_tape(self):
    """Pushes a new tape onto the tape stack."""
    if self._recording:
      raise ValueError("Tape is still recording, This can happen if you try to "
                       "re-enter an already-active tape.")
    if self._tape is None:
      self._tape = tape.push_new_tape(
          persistent=self._persistent,
          watch_accessed_variables=self._watch_accessed_variables)
    else:
      tape.push_tape(self._tape)
    self._recording = True
  def _pop_tape(self):
    if not self._recording:
      raise ValueError("Tape is not recording.")
    tape.pop_tape(self._tape)
    self._recording = False
  @tf_contextlib.contextmanager
  def _ensure_recording(self):
    """Ensures that this tape is recording."""
    if not self._recording:
      try:
        self._push_tape()
        yield
      finally:
        self._pop_tape()
    else:
      yield
  # TODO(b/209081027): Add a variable in composite tensor test case after
  # variables become composite tensors.
  def watch(self, tensor):
    """Ensures that `tensor` is being traced by this tape.
    Args:
      tensor: a Tensor/Variable or list of Tensors/Variables.
    Raises:
      ValueError: if it encounters something that is not a tensor.
    """
    for t in _extract_tensors_and_variables(tensor):
      if not backprop_util.IsTrainable(t):
        logging.log_first_n(
            logging.WARN, "The dtype of the watched tensor must be "
            "floating (e.g. tf.float32), got %r", 5, t.dtype)
      if hasattr(t, "handle"):
        # There are many variable-like objects, all of them currently have
        # `handle` attribute that points to a tensor. If this changes,
        # internals of watch_variable need to change as well.
        tape.watch_variable(self._tape, t)
      else:
        tape.watch(self._tape, t)
  @tf_contextlib.contextmanager
  def stop_recording(self):
    """Temporarily stops recording operations on this tape.
    Operations executed while this context manager is active will not be
    recorded on the tape. This is useful for reducing the memory used by tracing
    all computations.
    For example:
    >>> x = tf.constant(4.0)
    >>> with tf.GradientTape() as tape:
    ...   with tape.stop_recording():
    ...     y = x ** 2
    >>> dy_dx = tape.gradient(y, x)
    >>> print(dy_dx)
    None
    Yields:
      None
    Raises:
      RuntimeError: if the tape is not currently recording.
    """
    if self._tape is None:
      raise RuntimeError(
          "Trying to stop recording a tape which is not recording.")
    self._pop_tape()
    try:
      yield
    finally:
      self._push_tape()
  def reset(self):
    """Clears all information stored in this tape.
    Equivalent to exiting and reentering the tape context manager with a new
    tape. For example, the two following code blocks are equivalent:
    ```
    with tf.GradientTape() as t:
      loss = loss_fn()
    with tf.GradientTape() as t:
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
    # The following is equivalent to the above
    with tf.GradientTape() as t:
      loss = loss_fn()
      t.reset()
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
    ```
    This is useful if you don't want to exit the context manager for the tape,
    or can't because the desired reset point is inside a control flow construct:
    ```
    with tf.GradientTape() as t:
      loss = ...
      if loss > k:
        t.reset()
    ```
    """
    self._pop_tape()
    self._tape = None
    self._push_tape()
  def watched_variables(self):
    """Returns variables watched by this tape in order of construction."""
    if self._tape is not None:
      self._watched_variables = self._tape.watched_variables()
    return self._watched_variables
  def gradient(self,
               target,
               sources,
               output_gradients=None,
               unconnected_gradients=UnconnectedGradients.NONE):
    """Computes the gradient using operations recorded in context of this tape.
    Note: Unless you set `persistent=True` a GradientTape can only be used to
    compute one set of gradients (or jacobians).
    In addition to Tensors, gradient also supports RaggedTensors. For example,
    >>> x = tf.ragged.constant([[1.0, 2.0], [3.0]])
    >>> with tf.GradientTape() as g:
    ...   g.watch(x)
    ...   y = x * x
    >>> g.gradient(y, x)
    <tf.RaggedTensor [[2.0, 4.0], [6.0]]>
    Args:
      target: a list or nested structure of Tensors or Variables or
        CompositeTensors to be differentiated.
      sources: a list or nested structure of Tensors or Variables or
        CompositeTensors. `target` will be differentiated against elements in
        `sources`.
      output_gradients: a list of gradients, one for each differentiable
        element of target. Defaults to None.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None, or
      CompositeTensor), one for each element in `sources`. Returned structure
      is the same as the structure of `sources`.
    Raises:
      RuntimeError: If called on a used, non-persistent tape.
      RuntimeError: If called inside the context of the tape.
      TypeError: If the target is a None object.
      ValueError: If the target is a variable or if unconnected gradients is
       called with an unknown value.
    """
    if self._tape is None:
      raise RuntimeError("A non-persistent GradientTape can only be used to "
                         "compute one set of gradients (or jacobians)")
    if self._recording:
      if not self._persistent:
        self._pop_tape()
      else:
        logging.log_first_n(
            logging.WARN, "Calling GradientTape.gradient on a persistent "
            "tape inside its context is significantly less "
            "efficient than calling it outside the context (it "
            "causes the gradient ops to be recorded on the "
            "tape, leading to increased CPU and memory usage). "
            "Only call GradientTape.gradient inside the "
            "context if you actually want to trace the "
            "gradient in order to compute higher order "
            "derivatives.", 1)
    if target is None:
      raise TypeError("Argument `target` should be a list or nested structure"
                      " of Tensors, Variables or CompositeTensors to be "
                      "differentiated, but received None.")
    flat_targets = []
    for t in nest.flatten(target):
      flat_targets.append(_handle_or_self(t))
    flat_targets = composite_tensor_gradient.get_flat_tensors_for_gradients(
        flat_targets)
    for t in flat_targets:
      if not backprop_util.IsTrainable(t):
        logging.vlog(
            1, "The dtype of the target tensor must be "
            "floating (e.g. tf.float32) when calling GradientTape.gradient, "
            "got %r", t.dtype)
    flat_sources_raw = nest.flatten(sources)
    flat_sources = []
    for t in flat_sources_raw:
      flat_sources.append(_handle_or_self(t))
    flat_sources = composite_tensor_gradient.get_flat_tensors_for_gradients(
        flat_sources)
    for t in flat_sources:
      if not backprop_util.IsTrainable(t):
        logging.vlog(
            1, "The dtype of the source tensor must be "
            "floating (e.g. tf.float32) when calling GradientTape.gradient, "
            "got %r", t.dtype)
      if getattr(t, "is_packed", False):
        raise ValueError(
            "GradientTape.gradient is not supported on packed EagerTensors yet."
        )
    if output_gradients is not None:
      output_gradients = nest.flatten(
          variable_utils.convert_variables_to_tensors(output_gradients))
      output_gradients = (
          composite_tensor_gradient.get_flat_tensors_for_gradients(
              output_gradients))
      output_gradients = [None if x is None else ops.convert_to_tensor(x)
                          for x in output_gradients]
    flat_grad = imperative_grad.imperative_grad(
        self._tape,
        flat_targets,
        flat_sources,
        output_gradients=output_gradients,
        sources_raw=flat_sources_raw,
        unconnected_gradients=unconnected_gradients)
    if not self._persistent:
      # Keep track of watched variables before setting tape to None
      self._watched_variables = self._tape.watched_variables()
      self._tape = None
    flat_sources_raw = nest.map_structure(_handle_or_self, flat_sources_raw)
    flat_grad = composite_tensor_gradient.replace_flat_tensors_for_gradients(
        flat_sources_raw, flat_grad)
    grad = nest.pack_sequence_as(sources, flat_grad)
    return grad
  def jacobian(self,
               target,
               sources,
               unconnected_gradients=UnconnectedGradients.NONE,
               parallel_iterations=None,
               experimental_use_pfor=True):
    """Computes the jacobian using operations recorded in context of this tape.
    Note: Unless you set `persistent=True` a GradientTape can only be used to
    compute one set of gradients (or jacobians).
    Note: By default the jacobian implementation uses parallel for (pfor), which
    creates a tf.function under the hood for each jacobian call. For better
    performance, and to avoid recompilation and vectorization rewrites on each
    call, enclose GradientTape code in @tf.function.
    See[wikipedia
    article](http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant)
    for the definition of a Jacobian.
    Example usage:
    ```python
    with tf.GradientTape() as g:
      x  = tf.constant([1.0, 2.0])
      g.watch(x)
      y = x * x
    jacobian = g.jacobian(y, x)
    # jacobian value is [[2., 0.], [0., 4.]]
    ```
    Args:
      target: Tensor to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, vectorizes the jacobian computation. Else
        falls back to a sequential while_loop. Vectorization can sometimes fail
        or lead to excessive memory usage. This option can be used to disable
        vectorization in such cases.
    Returns:
      A list or nested structure of Tensors (or None), one for each element in
      `sources`. Returned structure is the same as the structure of `sources`.
      Note if any gradient is sparse (IndexedSlices), jacobian function
      currently makes it dense and returns a Tensor instead. This may change in
      the future.
    Raises:
      RuntimeError: If called on a used, non-persistent tape.
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails.
    """
    if self._tape is None:
      raise RuntimeError("A non-persistent GradientTape can only be used to "
                         "compute one set of gradients (or jacobians)")
    flat_sources = nest.flatten(sources)
    target_static_shape = target.shape
    target_shape = array_ops.shape(target)
    # Note that we push and pop the tape here and below. This is needed since we
    # need gradients through the enclosed operations.
    with self._ensure_recording():
      target = array_ops.reshape(target, [-1])
    def loop_fn(i):
      with self._ensure_recording():
        y = array_ops.gather(target, i)
      return self.gradient(y, flat_sources,
                           unconnected_gradients=unconnected_gradients)
    try:
      target_size = int(target.shape[0])
    except TypeError:
      target_size = array_ops.shape(target)[0]
    if experimental_use_pfor:
      try:
        output = pfor_ops.pfor(loop_fn, target_size,
                               parallel_iterations=parallel_iterations)
      except ValueError as err:
        raise ValueError(
            "Encountered an exception while vectorizing the "
            "jacobian computation. Vectorization can be disabled by setting"
            " experimental_use_pfor to False.") from err
    else:
      if context.executing_eagerly() and not self._persistent:
        raise RuntimeError(
            "GradientTape must be created with persistent=True"
            " to compute the jacobian with eager execution enabled and with "
            " experimental_use_pfor set to False.")
      output = pfor_ops.for_loop(
          loop_fn, [target.dtype] * len(flat_sources), target_size,
          parallel_iterations=parallel_iterations)
    for i, out in enumerate(output):
      if out is not None:
        new_shape = array_ops.concat(
            [target_shape, array_ops.shape(out)[1:]], axis=0)
        out = array_ops.reshape(out, new_shape)
        if context.executing_eagerly():
          out.set_shape(target_static_shape.concatenate(flat_sources[i].shape))
      output[i] = out
    return nest.pack_sequence_as(sources, output)
  def batch_jacobian(self,
                     target,
                     source,
                     unconnected_gradients=UnconnectedGradients.NONE,
                     parallel_iterations=None,
                     experimental_use_pfor=True):
    """Computes and stacks per-example jacobians.
    See [wikipedia article](http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant)
    for the definition of a Jacobian. This function is essentially an efficient
    implementation of the following:
    `tf.stack([self.jacobian(y[i], x[i]) for i in range(x.shape[0])])`.
    Note that compared to `GradientTape.jacobian` which computes gradient of
    each output value w.r.t each input value, this function is useful when
    `target[i,...]` is independent of `source[j,...]` for `j != i`. This
    assumption allows more efficient computation as compared to
    `GradientTape.jacobian`. The output, as well as intermediate activations,
    are lower dimensional and avoid a bunch of redundant zeros which would
    result in the jacobian computation given the independence assumption.
    Note: Unless you set `persistent=True` a GradientTape can only be used to
    compute one set of gradients (or jacobians).
    Note: By default the batch_jacobian implementation uses parallel for (pfor),
    which creates a tf.function under the hood for each batch_jacobian call.
    For better performance, and to avoid recompilation and vectorization
    rewrites on each call, enclose GradientTape code in @tf.function.
    Example usage:
    ```python
    with tf.GradientTape() as g:
      x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
      g.watch(x)
      y = x * x
    batch_jacobian = g.batch_jacobian(y, x)
    # batch_jacobian is [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]
    ```
    Args:
      target: A tensor with rank 2 or higher and with shape [b, y1, ..., y_n].
        `target[i,...]` should only depend on `source[i,...]`.
      source: A tensor with rank 2 or higher and with shape [b, x1, ..., x_m].
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, uses pfor for computing the Jacobian. Else
        uses a tf.while_loop.
    Returns:
      A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
      is the jacobian of `target[i, ...]` w.r.t. `source[i, ...]`, i.e. stacked
      per-example jacobians.
    Raises:
      RuntimeError: If called on a used, non-persistent tape.
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails or if first
        dimension of `target` and `source` do not match.
    """
    if self._tape is None:
      raise RuntimeError("A non-persistent GradientTape can only be used to"
                         "compute one set of gradients (or jacobians)")
    target_shape = target.shape
    if target_shape.rank is None:
      dim = tensor_shape.Dimension(None)
    else:
      dim = target_shape.dims[0]
    if not (target_shape.with_rank_at_least(2) and
            source.shape.with_rank_at_least(2) and
            dim.is_compatible_with(source.shape[0])):
      raise ValueError(
          "Need first dimension of target shape (%s) and "
          "source shape (%s) to match." % (target.shape, source.shape))
    if target_shape.is_fully_defined():
      batch_size = int(target_shape[0])
      target_row_size = target_shape.num_elements() // batch_size
    else:
      target_shape = array_ops.shape(target)
      batch_size = target_shape[0]
      target_row_size = array_ops.size(target) // batch_size
    source_shape = array_ops.shape(source)
    # Flatten target to 2-D.
    # Note that we push and pop the tape here and below. This is needed since we
    # need gradients through the enclosed operations.
    with self._ensure_recording():
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size, source_shape[0])]):
        target = array_ops.reshape(target, [batch_size, target_row_size])
    run_once = False
    def loop_fn(i):
      nonlocal run_once
      if run_once and not self._persistent:
        if parallel_iterations is not None:
          raise RuntimeError(
              "GradientTape must be created with persistent=True"
              " to compute the batch_jacobian with parallel_iterations.")
        else:
          raise RuntimeError(
              "GradientTape must be created with persistent=True"
              " to compute the batch_jacobian.")
      run_once = True
      with self._ensure_recording():
        y = array_ops.gather(target, i, axis=1)
      return self.gradient(y, source,
                           unconnected_gradients=unconnected_gradients)
    if experimental_use_pfor:
      try:
        output = pfor_ops.pfor(loop_fn, target_row_size,
                               parallel_iterations=parallel_iterations)
      except ValueError as err:
        raise ValueError(
            "Encountered an exception while vectorizing the "
            "batch_jacobian computation. Vectorization can be disabled by "
            "setting experimental_use_pfor to False.") from err
    else:
      if context.executing_eagerly() and not self._persistent:
        raise RuntimeError(
            "GradientTape must be created with persistent=True"
            " to compute the batch_jacobian with eager execution enabled and "
            " with experimental_use_pfor set to False.")
      output = pfor_ops.for_loop(loop_fn, target.dtype, target_row_size,
                                 parallel_iterations=parallel_iterations)
    new_shape = array_ops.concat([target_shape, source_shape[1:]], axis=0)
    if output is None:
      # Note that this block is returning zeros when it could use `None` to
      # represent unconnected gradients. This is to maintain compatibility with
      # the previous behavior, which ignored `unconnected_gradients`.
      output = array_ops.zeros(new_shape, target.dtype)
      return output
    else:
      output = array_ops.reshape(output,
                                 [target_row_size, batch_size, -1])
      output = array_ops.transpose(output, [1, 0, 2])
      output = array_ops.reshape(output, new_shape)
      return output
