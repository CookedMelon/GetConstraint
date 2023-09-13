@tf_export("strings.unicode_split_with_offsets")
@dispatch.add_dispatch_support
def unicode_split_with_offsets(input,
                               input_encoding,
                               errors="replace",
                               replacement_char=0xFFFD,
                               name=None):
  r"""Splits each string into a sequence of code points with start offsets.
  This op is similar to `tf.strings.decode(...)`, but it also returns the
  start offset for each character in its respective string.  This information
  can be used to align the characters with the original byte sequence.
  Returns a tuple `(chars, start_offsets)` where:
  * `chars[i1...iN, j]` is the substring of `input[i1...iN]` that encodes its
    `j`th character, when decoded using `input_encoding`.
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
      substrings in `input` when `errors='replace'`.
    name: A name for the operation (optional).
  Returns:
    A tuple of `N+1` dimensional tensors `(codepoints, start_offsets)`.
    * `codepoints` is an `int32` tensor with shape `[D1...DN, (num_chars)]`.
    * `offsets` is an `int64` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensors are `tf.Tensor`s if `input` is a scalar, or
    `tf.RaggedTensor`s otherwise.
  #### Example:
  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> result = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
  >>> result[0].to_list()  # character substrings
  [[b'G', b'\xc3\xb6', b'\xc3\xb6', b'd', b'n', b'i', b'g', b'h', b't'],
   [b'\xf0\x9f\x98\x8a']]
  >>> result[1].to_list()  # offsets
  [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]]
  """
  with ops.name_scope(name, "UnicodeSplitWithOffsets", [input]):
    codepoints, offsets = _unicode_decode(input, input_encoding, errors,
                                          replacement_char, False,
                                          with_offsets=True)
    chars = unicode_encode(
        ragged_array_ops.expand_dims(codepoints, -1),
        output_encoding=input_encoding,
        errors=errors,
        replacement_char=replacement_char)
    return chars, offsets
