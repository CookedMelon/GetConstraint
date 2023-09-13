@tf_export("print")
@dispatch.add_dispatch_support
def print_v2(*inputs, **kwargs):
  """Print the specified inputs.
  A TensorFlow operator that prints the specified inputs to a desired
  output stream or logging level. The inputs may be dense or sparse Tensors,
  primitive python objects, data structures that contain tensors, and printable
  Python objects. Printed tensors will recursively show the first and last
  elements of each dimension to summarize.
  Example:
    Single-input usage:
    ```python
    tensor = tf.range(10)
    tf.print(tensor, output_stream=sys.stderr)
    ```
    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)
    Multi-input usage:
    ```python
    tensor = tf.range(10)
    tf.print("tensors:", tensor, {2: tensor * 2}, output_stream=sys.stdout)
    ```
    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to
    sys.stdout)
    Changing the input separator:
    ```python
    tensor_a = tf.range(2)
    tensor_b = tensor_a * 2
    tf.print(tensor_a, tensor_b, output_stream=sys.stderr, sep=',')
    ```
    (This prints "[0 1],[0 2]" to sys.stderr)
    Usage in a `tf.function`:
    ```python
    @tf.function
    def f():
        tensor = tf.range(10)
        tf.print(tensor, output_stream=sys.stderr)
        return tensor
    range_tensor = f()
    ```
    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)
  *Compatibility usage in TF 1.x graphs*:
    In graphs manually created outside of `tf.function`, this method returns
    the created TF operator that prints the data. To make sure the
    operator runs, users need to pass the produced op to
    `tf.compat.v1.Session`'s run method, or to use the op as a control
    dependency for executed ops by specifying
    `with tf.compat.v1.control_dependencies([print_op])`.
    ```python
    tf.compat.v1.disable_v2_behavior()  # for TF1 compatibility only
    sess = tf.compat.v1.Session()
    with sess.as_default():
      tensor = tf.range(10)
      print_op = tf.print("tensors:", tensor, {2: tensor * 2},
                          output_stream=sys.stdout)
      with tf.control_dependencies([print_op]):
        tripled_tensor = tensor * 3
      sess.run(tripled_tensor)
    ```
    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to
    sys.stdout)
  Note: In Jupyter notebooks and colabs, `tf.print` prints to the notebook
    cell outputs. It will not write to the notebook kernel's console logs.
  Args:
    *inputs: Positional arguments that are the inputs to print. Inputs in the
      printed output will be separated by spaces. Inputs may be python
      primitives, tensors, data structures such as dicts and lists that may
      contain tensors (with the data structures possibly nested in arbitrary
      ways), and printable python objects.
    output_stream: The output stream, logging level, or file to print to.
      Defaults to sys.stderr, but sys.stdout, tf.compat.v1.logging.info,
      tf.compat.v1.logging.warning, tf.compat.v1.logging.error,
      absl.logging.info, absl.logging.warning and absl.logging.error are also
      supported. To print to a file, pass a string started with "file://"
      followed by the file path, e.g., "file:///tmp/foo.out".
    summarize: The first and last `summarize` elements within each dimension are
      recursively printed per Tensor. If None, then the first 3 and last 3
      elements of each dimension are printed for each tensor. If set to -1, it
      will print all elements of every tensor.
    sep: The string to use to separate the inputs. Defaults to " ".
    end: End character that is appended at the end the printed string. Defaults
      to the newline character.
    name: A name for the operation (optional).
  Returns:
    None when executing eagerly. During graph tracing this returns
    a TF operator that prints the specified inputs in the specified output
    stream or logging level. This operator will be automatically executed
    except inside of `tf.compat.v1` graphs and sessions.
  Raises:
    ValueError: If an unsupported output stream is specified.
  """
  # Because we are using arbitrary-length positional arguments, python 2
  # does not support explicitly specifying the keyword arguments in the
  # function definition. So, we manually get the keyword arguments w/ default
  # values here.
  output_stream = kwargs.pop("output_stream", sys.stderr)
  name = kwargs.pop("name", None)
  summarize = kwargs.pop("summarize", 3)
  sep = kwargs.pop("sep", " ")
  end = kwargs.pop("end", os.linesep)
  if kwargs:
    raise ValueError("Unrecognized keyword arguments for tf.print: %s" % kwargs)
  format_name = None
  if name:
    format_name = name + "_format"
  # Match the C++ string constants representing the different output streams.
  # Keep this updated!
  output_stream_to_constant = {
      sys.stdout: "stdout",
      sys.stderr: "stderr",
      tf_logging.INFO: "log(info)",
      tf_logging.info: "log(info)",
      tf_logging.WARN: "log(warning)",
      tf_logging.warning: "log(warning)",
      tf_logging.warn: "log(warning)",
      tf_logging.ERROR: "log(error)",
      tf_logging.error: "log(error)",
      logging.INFO: "log(info)",
      logging.info: "log(info)",
      logging.INFO: "log(info)",
      logging.WARNING: "log(warning)",
      logging.WARN: "log(warning)",
      logging.warning: "log(warning)",
      logging.warn: "log(warning)",
      logging.ERROR: "log(error)",
      logging.error: "log(error)",
  }
  if _is_filepath(output_stream):
    output_stream_string = output_stream
  else:
    output_stream_string = output_stream_to_constant.get(output_stream)
    if not output_stream_string:
      raise ValueError("Unsupported output stream, logging level, or file." +
                       str(output_stream) +
                       ". Supported streams are sys.stdout, "
                       "sys.stderr, tf.logging.info, "
                       "tf.logging.warning, tf.logging.error. " +
                       "File needs to be in the form of 'file://<filepath>'.")
  # If we are only printing a single string scalar, there is no need to format
  if (len(inputs) == 1 and tensor_util.is_tf_type(inputs[0]) and
      (not isinstance(inputs[0], sparse_tensor.SparseTensor)) and
      (inputs[0].shape.ndims == 0) and (inputs[0].dtype == dtypes.string)):
    formatted_string = inputs[0]
  # Otherwise, we construct an appropriate template for the tensors we are
  # printing, and format the template using those tensors.
  else:
    # For each input to this print function, we extract any nested tensors,
    # and construct an appropriate template to format representing the
    # printed input.
    templates = []
    tensors = []
    # If an input to the print function is of type `OrderedDict`, sort its
    # elements by the keys for consistency with the ordering of `nest.flatten`.
    # This is not needed for `dict` types because `pprint.pformat()` takes care
    # of printing the template in a sorted fashion.
    inputs_ordered_dicts_sorted = []
    for input_ in inputs:
      if isinstance(input_, py_collections.OrderedDict):
        inputs_ordered_dicts_sorted.append(
            py_collections.OrderedDict(sorted(input_.items())))
      else:
        inputs_ordered_dicts_sorted.append(input_)
    tensor_free_structure = nest.map_structure(
        lambda x: "" if tensor_util.is_tf_type(x) else x,
        inputs_ordered_dicts_sorted)
    tensor_free_template = " ".join(
        pprint.pformat(x) for x in tensor_free_structure)
    placeholder = _generate_placeholder_string(tensor_free_template)
    for input_ in inputs:
      placeholders = []
      # Use the nest utilities to flatten & process any nested elements in this
      # input. The placeholder for a tensor in the template should be the
      # placeholder string, and the placeholder for a non-tensor can just be
      # the printed value of the non-tensor itself.
      for x in nest.flatten(input_):
        # support sparse tensors
        if isinstance(x, sparse_tensor.SparseTensor):
          tensors.extend([x.indices, x.values, x.dense_shape])
          placeholders.append(
              "SparseTensor(indices={}, values={}, shape={})".format(
                  placeholder, placeholder, placeholder))
        elif tensor_util.is_tf_type(x):
          tensors.append(x)
          placeholders.append(placeholder)
        else:
          placeholders.append(x)
      if isinstance(input_, str):
        # If the current input to format/print is a normal string, that string
        # can act as the template.
        cur_template = input_
      else:
        # We pack the placeholders into a data structure that matches the
        # input data structure format, then format that data structure
        # into a string template.
        #
        # NOTE: We must use pprint.pformat here for building the template for
        # unordered data structures such as `dict`, because `str` doesn't
        # guarantee orderings, while pprint prints in sorted order. pprint
        # will match the ordering of `nest.flatten`.
        # This even works when nest.flatten reorders OrderedDicts, because
        # pprint is printing *after* the OrderedDicts have been reordered.
        cur_template = pprint.pformat(
            nest.pack_sequence_as(input_, placeholders))
      templates.append(cur_template)
    # We join the templates for the various inputs into a single larger
    # template. We also remove all quotes surrounding the placeholders, so that
    # the formatted/printed output will not contain quotes around tensors.
    # (example of where these quotes might appear: if we have added a
    # placeholder string into a list, then pretty-formatted that list)
    template = sep.join(templates)
    template = template.replace("'" + placeholder + "'", placeholder)
    formatted_string = string_ops.string_format(
        inputs=tensors,
        template=template,
        placeholder=placeholder,
        summarize=summarize,
        name=format_name)
  return gen_logging_ops.print_v2(
      formatted_string, output_stream=output_stream_string, name=name, end=end)
