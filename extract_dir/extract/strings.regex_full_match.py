@tf_export("strings.regex_full_match")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def regex_full_match(input, pattern, name=None):
  r"""Match elements of `input` with regex `pattern`.
  Args:
    input: string `Tensor`, the source strings to process.
    pattern: string or scalar string `Tensor`, regular expression to use,
      see more details at https://github.com/google/re2/wiki/Syntax
    name: Name of the op.
  Returns:
    bool `Tensor` of the same shape as `input` with match results.
  """
  if isinstance(pattern, util_compat.bytes_or_text_types):
    # When `pattern` is static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_full_match(
        input=input, pattern=pattern, name=name)
  return gen_string_ops.regex_full_match(
      input=input, pattern=pattern, name=name)
