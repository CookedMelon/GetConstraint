@tf_export("strings.ngrams")
@dispatch.add_dispatch_support
def ngrams(data,
           ngram_width,
           separator=" ",
           pad_values=None,
           padding_width=None,
           preserve_short_sequences=False,
           name=None):
  """Create a tensor of n-grams based on `data`.
  Creates a tensor of n-grams based on `data`. The n-grams are created by
  joining windows of `width` adjacent strings from the inner axis of `data`
  using `separator`.
  The input data can be padded on both the start and end of the sequence, if
  desired, using the `pad_values` argument. If set, `pad_values` should contain
  either a tuple of strings or a single string; the 0th element of the tuple
  will be used to pad the left side of the sequence and the 1st element of the
  tuple will be used to pad the right side of the sequence. The `padding_width`
  arg controls how many padding values are added to each side; it defaults to
  `ngram_width-1`.
  If this op is configured to not have padding, or if it is configured to add
  padding with `padding_width` set to less than ngram_width-1, it is possible
  that a sequence, or a sequence plus padding, is smaller than the ngram
  width. In that case, no ngrams will be generated for that sequence. This can
  be prevented by setting `preserve_short_sequences`, which will cause the op
  to always generate at least one ngram per non-empty sequence.
  Examples:
  >>> tf.strings.ngrams(["A", "B", "C", "D"], 2).numpy()
  array([b'A B', b'B C', b'C D'], dtype=object)
  >>> tf.strings.ngrams(["TF", "and", "keras"], 1).numpy()
  array([b'TF', b'and', b'keras'], dtype=object)
  Args:
    data: A Tensor or RaggedTensor containing the source data for the ngrams.
    ngram_width: The width(s) of the ngrams to create. If this is a list or
      tuple, the op will return ngrams of all specified arities in list order.
      Values must be non-Tensor integers greater than 0.
    separator: The separator string used between ngram elements. Must be a
      string constant, not a Tensor.
    pad_values: A tuple of (left_pad_value, right_pad_value), a single string,
      or None. If None, no padding will be added; if a single string, then that
      string will be used for both left and right padding. Values must be Python
      strings.
    padding_width: If set, `padding_width` pad values will be added to both
      sides of each sequence. Defaults to `ngram_width`-1. Must be greater than
      0. (Note that 1-grams are never padded, regardless of this value.)
    preserve_short_sequences: If true, then ensure that at least one ngram is
      generated for each input sequence.  In particular, if an input sequence is
      shorter than `min(ngram_width) + 2*pad_width`, then generate a single
      ngram containing the entire sequence.  If false, then no ngrams are
      generated for these short input sequences.
    name: The op name.
  Returns:
    A RaggedTensor of ngrams. If `data.shape=[D1...DN, S]`, then
    `output.shape=[D1...DN, NUM_NGRAMS]`, where
    `NUM_NGRAMS=S-ngram_width+1+2*padding_width`.
  Raises:
    TypeError: if `pad_values` is set to an invalid type.
    ValueError: if `pad_values`, `padding_width`, or `ngram_width` is set to an
      invalid value.
  """
  with ops.name_scope(name, "StringNGrams", [data]):
    if pad_values is None:
      left_pad = ""
      right_pad = ""
    elif isinstance(pad_values, (list, tuple)):
      if (not isinstance(pad_values[0], util_compat.bytes_or_text_types) or
          not isinstance(pad_values[1], util_compat.bytes_or_text_types)):
        raise TypeError(
            "pad_values must be a string, tuple of strings, or None.")
      left_pad = pad_values[0]
      right_pad = pad_values[1]
    else:
      if not isinstance(pad_values, util_compat.bytes_or_text_types):
        raise TypeError(
            "pad_values must be a string, tuple of strings, or None.")
      left_pad = pad_values
      right_pad = pad_values
    if padding_width is not None and padding_width < 1:
      raise ValueError("padding_width must be greater than 0.")
    if padding_width is not None and pad_values is None:
      raise ValueError("pad_values must be provided if padding_width is set.")
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        data, name="data", dtype=dtypes.string)
    # preserve the shape of the data if it is a tensor
    to_tensor = False
    if isinstance(data, ops.Tensor):
      dense_shape = array_ops.concat([array_ops.shape(data)[:-1], [-1]], axis=0)
      to_tensor = True
    if not isinstance(data, ragged_tensor.RaggedTensor):
      if data.shape.ndims is None:
        raise ValueError("Rank of data must be known.")
      elif data.shape.ndims == 0:
        raise ValueError("Data must have rank>0")
      elif data.shape.ndims == 1:
        rt = ragged_tensor.RaggedTensor.from_row_starts(
            data, [0], validate=False)
        return ngrams(rt, ngram_width, separator, pad_values, padding_width,
                      preserve_short_sequences, name)[0]
      else:
        data = ragged_tensor.RaggedTensor.from_tensor(
            data, ragged_rank=data.shape.ndims - 1)
    if data.ragged_rank > 1:
      output = data.with_values(
          ngrams(data.values, ngram_width, separator, pad_values, padding_width,
                 preserve_short_sequences, name))
      return array_ops.reshape(output.flat_values,
                               dense_shape) if to_tensor else output
    if pad_values is None:
      padding_width = 0
    if pad_values is not None and padding_width is None:
      padding_width = -1
    if not isinstance(ngram_width, (list, tuple)):
      ngram_widths = [ngram_width]
    else:
      ngram_widths = ngram_width
    for width in ngram_widths:
      if width < 1:
        raise ValueError("All ngram_widths must be greater than 0. Got %s" %
                         ngram_width)
    output, output_splits = gen_string_ops.string_n_grams(
        data=data.flat_values,
        data_splits=data.row_splits,
        separator=separator,
        ngram_widths=ngram_widths,
        left_pad=left_pad,
        right_pad=right_pad,
        pad_width=padding_width,
        preserve_short_sequences=preserve_short_sequences)
    # if the input is Dense tensor, the output should also be a dense tensor
    output = ragged_tensor.RaggedTensor.from_row_splits(
        values=output, row_splits=output_splits, validate=False)
    return array_ops.reshape(output.flat_values,
                             dense_shape) if to_tensor else output
