@tf_export("strings.unicode_decode_with_offsets")
@dispatch.add_dispatch_support
def unicode_decode_with_offsets(input,
                                input_encoding,
                                errors="replace",
                                replacement_char=0xFFFD,
                                replace_control_characters=False,
                                name=None):
  r"""Decodes each string into a sequence of code points with start offsets.
  This op is similar to `tf.strings.decode(...)`, but it also returns the
  start offset for each character in its respective string.  This information
  can be used to align the characters with the original byte sequence.
  Returns a tuple `(codepoints, start_offsets)` where:
  * `codepoints[i1...iN, j]` is the Unicode codepoint for the `j`th character
    in `input[i1...iN]`, when decoded using `input_encoding`.
  * `start_offsets[i1...iN, j]` is the start byte offset for the `j`th
    character in `input[i1...iN]`, when decoded using `input_encoding`.
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
    A tuple of `N+1` dimensional tensors `(codepoints, start_offsets)`.
    * `codepoints` is an `int32` tensor with shape `[D1...DN, (num_chars)]`.
    * `offsets` is an `int64` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensors are `tf.Tensor`s if `input` is a scalar, or
    `tf.RaggedTensor`s otherwise.
  #### Example:
  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> result = tf.strings.unicode_decode_with_offsets(input, 'UTF-8')
  >>> result[0].to_list()  # codepoints
  [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]]
  >>> result[1].to_list()  # offsets
  [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]]
  """
  with ops.name_scope(name, "UnicodeDecodeWithOffsets", [input]):
    return _unicode_decode(input, input_encoding, errors, replacement_char,
                           replace_control_characters, with_offsets=True)
