@tf_export("strings.unicode_decode")
@dispatch.add_dispatch_support
def unicode_decode(input,
                   input_encoding,
                   errors="replace",
                   replacement_char=0xFFFD,
                   replace_control_characters=False,
                   name=None):
  r"""Decodes each string in `input` into a sequence of Unicode code points.
  `result[i1...iN, j]` is the Unicode codepoint for the `j`th character in
  `input[i1...iN]`, when decoded using `input_encoding`.
  Args:
    input: An `N` dimensional potentially ragged `string` tensor with shape
      `[D1...DN]`.  `N` must be statically known.
    input_encoding: String name for the unicode encoding that should be used to
      decode each string.
    errors: Specifies the response when an input string can't be converted
      using the indicated encoding. One of:
      * `'strict'`: Raise an exception for any illegal substrings.
      * `'replace'`: Replace illegal substrings with `replacement_char`.
      * `'ignore'`: Skip illegal substrings.
    replacement_char: The replacement codepoint to be used in place of invalid
      substrings in `input` when `errors='replace'`; and in place of C0 control
      characters in `input` when `replace_control_characters=True`.
    replace_control_characters: Whether to replace the C0 control characters
      `(U+0000 - U+001F)` with the `replacement_char`.
    name: A name for the operation (optional).
  Returns:
    A `N+1` dimensional `int32` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensor is a `tf.Tensor` if `input` is a scalar, or a
    `tf.RaggedTensor` otherwise.
  #### Example:
  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> tf.strings.unicode_decode(input, 'UTF-8').to_list()
  [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]]
  """
  with ops.name_scope(name, "UnicodeDecode", [input]):
    return _unicode_decode(input, input_encoding, errors, replacement_char,
                           replace_control_characters, with_offsets=False)
