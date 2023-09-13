@tf_export("nn.ctc_beam_search_decoder", v1=["nn.ctc_beam_search_decoder_v2"])
@dispatch.add_dispatch_support
def ctc_beam_search_decoder_v2(inputs,
                               sequence_length,
                               beam_width=100,
                               top_paths=1):
  """Performs beam search decoding on the logits given in input.
  **Note** Although in general greedy search is a special case of beam-search
  with `top_paths=1` and `beam_width=1`, `ctc_beam_search_decoder` differs
  from `ctc_greedy_decoder` in the treatment of blanks when computing the
  probability of a sequence:
    - `ctc_beam_search_decoder` treats blanks as sequence termination
    - `ctc_greedy_decoder` treats blanks as regular elements
  Args:
    inputs: 3-D `float` `Tensor`, size `[max_time, batch_size, num_classes]`.
      The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths, having size
      `[batch_size]`.
    beam_width: An int scalar >= 0 (beam search beam width).
    top_paths: An int scalar >= 0, <= beam_width (controls output size).
  Returns:
    A tuple `(decoded, log_probabilities)` where
    decoded: A list of length top_paths, where `decoded[j]`
      is a `SparseTensor` containing the decoded outputs:
      `decoded[j].indices`: Indices matrix `[total_decoded_outputs[j], 2]`;
        The rows store: `[batch, time]`.
      `decoded[j].values`: Values vector, size `[total_decoded_outputs[j]]`.
        The vector stores the decoded classes for beam `j`.
      `decoded[j].dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length[j]]`.
    log_probability: A `float` matrix `[batch_size, top_paths]` containing
        sequence log-probabilities.
  """
  # Note, merge_repeated is an invalid optimization that is removed from the
  # public API: it returns low probability paths.
  return ctc_beam_search_decoder(
      inputs,
      sequence_length=sequence_length,
      beam_width=beam_width,
      top_paths=top_paths,
      merge_repeated=False)
