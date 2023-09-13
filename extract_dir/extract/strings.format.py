@tf_export("strings.format")
@dispatch.add_dispatch_support
def string_format(template, inputs, placeholder="{}", summarize=3, name=None):
  r"""Formats a string template using a list of tensors.
  Formats a string template using a list of tensors, abbreviating tensors by
  only printing the first and last `summarize` elements of each dimension
  (recursively). If formatting only one tensor into a template, the tensor does
  not have to be wrapped in a list.
  Example:
    Formatting a single-tensor template:
    >>> tensor = tf.range(5)
    >>> tf.strings.format("tensor: {}, suffix", tensor)
    <tf.Tensor: shape=(), dtype=string, numpy=b'tensor: [0 1 2 3 4], suffix'>
    Formatting a multi-tensor template:
    >>> tensor_a = tf.range(2)
    >>> tensor_b = tf.range(1, 4, 2)
    >>> tf.strings.format("a: {}, b: {}, suffix", (tensor_a, tensor_b))
    <tf.Tensor: shape=(), dtype=string, numpy=b'a: [0 1], b: [1 3], suffix'>
  Args:
    template: A string template to format tensor values into.
    inputs: A list of `Tensor` objects, or a single Tensor.
      The list of tensors to format into the template string. If a solitary
      tensor is passed in, the input tensor will automatically be wrapped as a
      list.
    placeholder: An optional `string`. Defaults to `{}`.
      At each placeholder occurring in the template, a subsequent tensor
      will be inserted.
    summarize: An optional `int`. Defaults to `3`.
      When formatting the tensors, show the first and last `summarize`
      entries of each tensor dimension (recursively). If set to -1, all
      elements of the tensor will be shown.
    name: A name for the operation (optional).
  Returns:
    A scalar `Tensor` of type `string`.
  Raises:
    ValueError: if the number of placeholders does not match the number of
      inputs.
  """
  # If there is only one tensor to format, we will automatically wrap it in a
  # list to simplify the user experience
  if tensor_util.is_tf_type(inputs):
    inputs = [inputs]
  if template.count(placeholder) != len(inputs):
    raise ValueError(f"The template expects {template.count(placeholder)} "
                     f"tensors, but the inputs only has {len(inputs)}. "
                     "Please ensure the number of placeholders in template "
                     "matches inputs length.")
  return gen_string_ops.string_format(inputs,
                                      template=template,
                                      placeholder=placeholder,
                                      summarize=summarize,
                                      name=name)
