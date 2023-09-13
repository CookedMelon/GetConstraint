@tf_export("strings.unicode_split")
@dispatch.add_dispatch_support
def unicode_split(input,
                  input_encoding,
                  errors="replace",
                  replacement_char=0xFFFD,
                  name=None):
  r"""Splits each string in `input` into a sequence of Unicode code points.
  `result[i1...iN, j]` is the substring of `input[i1...iN]` that encodes its
  `j`th character, when decoded using `input_encoding`.
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
      substrings in `input` when `errors='replace'`.
    name: A name for the operation (optional).
  Returns:
    A `N+1` dimensional `int32` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensor is a `tf.Tensor` if `input` is a scalar, or a
    `tf.RaggedTensor` otherwise.
  #### Example:
  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> tf.strings.unicode_split(input, 'UTF-8').to_list()
  [[b'G', b'\xc3\xb6', b'\xc3\xb6', b'd', b'n', b'i', b'g', b'h', b't'],
   [b'\xf0\x9f\x98\x8a']]
  """
  with ops.name_scope(name, "UnicodeSplit", [input]):
    codepoints = _unicode_decode(input, input_encoding, errors,
                                 replacement_char, False, with_offsets=False)
    return unicode_encode(
        ragged_array_ops.expand_dims(codepoints, -1),
        output_encoding=input_encoding,
        errors=errors,
        replacement_char=replacement_char)
