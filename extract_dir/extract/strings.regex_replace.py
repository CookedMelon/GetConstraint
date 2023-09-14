@tf_export(
    "strings.regex_replace", v1=["strings.regex_replace", "regex_replace"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("regex_replace")
def regex_replace(input, pattern, rewrite, replace_global=True, name=None):
  r"""Replace elements of `input` matching regex `pattern` with `rewrite`.
  >>> tf.strings.regex_replace("Text with tags.<br /><b>contains html</b>",
  ...                          "<[^>]+>", " ")
  <tf.Tensor: shape=(), dtype=string, numpy=b'Text with tags.  contains html '>
  Args:
    input: string `Tensor`, the source strings to process.
    pattern: string or scalar string `Tensor`, regular expression to use,
      see more details at https://github.com/google/re2/wiki/Syntax
    rewrite: string or scalar string `Tensor`, value to use in match
      replacement, supports backslash-escaped digits (\1 to \9) can be to insert
      text matching corresponding parenthesized group.
    replace_global: `bool`, if `True` replace all non-overlapping matches,
      else replace only the first match.
    name: A name for the operation (optional).
  Returns:
    string `Tensor` of the same shape as `input` with specified replacements.
  """
  if (isinstance(pattern, util_compat.bytes_or_text_types) and
      isinstance(rewrite, util_compat.bytes_or_text_types)):
    # When `pattern` and `rewrite` are static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_replace(
        input=input, pattern=pattern,
        rewrite=rewrite, replace_global=replace_global,
        name=name)
  return gen_string_ops.regex_replace(
      input=input, pattern=pattern,
      rewrite=rewrite, replace_global=replace_global,
      name=name)
