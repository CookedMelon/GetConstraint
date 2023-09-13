@tf_export("nn.ctc_loss", v1=[])
@dispatch.add_dispatch_support
def ctc_loss_v3(labels,
                logits,
                label_length,
                logit_length,
                logits_time_major=True,
                unique=None,
                blank_index=None,
                name=None):
  """Computes CTC (Connectionist Temporal Classification) loss.
  This op implements the CTC loss as presented in
  [Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
  Connectionist temporal classification (CTC) is a type of neural network output
  and associated scoring function, for training recurrent neural networks (RNNs)
  such as LSTM networks to tackle sequence problems where the timing is
  variable. It can be used for tasks like on-line handwriting recognition or
  recognizing phones in speech audio. CTC refers to the outputs and scoring, and
  is independent of the underlying neural network structure.
  Notes:
  - This class performs the softmax operation for you, so `logits` should be
    e.g. linear projections of outputs by an LSTM.
  - Outputs true repeated classes with blanks in between, and can also output
    repeated classes with no blanks in between that need to be collapsed by the
    decoder.
  - `labels` may be supplied as either a dense, zero-padded `Tensor` with a
    vector of label sequence lengths OR as a `SparseTensor`.
  - On TPU: Only dense padded `labels` are supported.
  - On CPU and GPU: Caller may use `SparseTensor` or dense padded `labels`
    but calling with a `SparseTensor` will be significantly faster.
  - Default blank label is `0` instead of `num_labels - 1` (where `num_labels`
    is the innermost dimension size of `logits`), unless overridden by
    `blank_index`.
  >>> tf.random.set_seed(50)
  >>> batch_size = 8
  >>> num_labels = 6
  >>> max_label_length = 5
  >>> num_frames = 12
  >>> labels = tf.random.uniform([batch_size, max_label_length],
  ...                            minval=1, maxval=num_labels, dtype=tf.int64)
  >>> logits = tf.random.uniform([num_frames, batch_size, num_labels])
  >>> label_length = tf.random.uniform([batch_size], minval=2,
  ...                                  maxval=max_label_length, dtype=tf.int64)
  >>> label_mask = tf.sequence_mask(label_length, maxlen=max_label_length,
  ...                               dtype=label_length.dtype)
  >>> labels *= label_mask
  >>> logit_length = [num_frames] * batch_size
  >>> with tf.GradientTape() as t:
  ...   t.watch(logits)
  ...   ref_loss = tf.nn.ctc_loss(
  ...       labels=labels,
  ...       logits=logits,
  ...       label_length=label_length,
  ...       logit_length=logit_length,
  ...       blank_index=0)
  >>> ref_grad = t.gradient(ref_loss, logits)
  Args:
    labels: `Tensor` of shape `[batch_size, max_label_seq_length]` or
      `SparseTensor`.
    logits: `Tensor` of shape `[frames, batch_size, num_labels]`. If
      `logits_time_major == False`, shape is `[batch_size, frames, num_labels]`.
    label_length: `Tensor` of shape `[batch_size]`. None, if `labels` is a
      `SparseTensor`. Length of reference label sequence in `labels`.
    logit_length: `Tensor` of shape `[batch_size]`. Length of input sequence in
      `logits`.
    logits_time_major: (optional) If True (default), `logits` is shaped [frames,
      batch_size, num_labels]. If False, shape is
      `[batch_size, frames, num_labels]`.
    unique: (optional) Unique label indices as computed by
      `ctc_unique_labels(labels)`.  If supplied, enable a faster, memory
      efficient implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from `num_labels`, ie, `-1` will reproduce the
      ctc_loss behavior of using `num_labels - 1` for the blank symbol. There is
      some memory/performance overhead to switching from the default of 0 as an
      additional shifted copy of `logits` may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".
  Returns:
    loss: A 1-D `float Tensor` of shape `[batch_size]`, containing negative log
    probabilities.
  Raises:
    ValueError: Argument `blank_index` must be provided when `labels` is a
    `SparseTensor`.
  References:
      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data
      with Recurrent Neural Networks:
        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)
        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))
      https://en.wikipedia.org/wiki/Connectionist_temporal_classification
  """
  if isinstance(labels, sparse_tensor.SparseTensor):
    if blank_index is None:
      raise ValueError(
          "Argument `blank_index` must be provided when `labels` is a "
          "`SparseTensor`.")
    if blank_index < 0:
      blank_index += _get_dim(logits, 2)
    logits = ops.convert_to_tensor(logits, name="logits")
    params = {
        "labels": labels,
        "logits": logits,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "blank_index": blank_index
    }
    if context.executing_eagerly():
      device_type = _get_context_device_type()
      can_use_gpu = (
          # Either user specified GPU or unspecified but GPU is available.
          (device_type == _GPU_DEVICE_NAME or
           (device_type is None and context.num_gpus() > 0)))
      # Under eager context, check the device placement and prefer the
      if can_use_gpu:
        res = _ctc_loss_op_cudnn(**params)
      else:
        res = _ctc_loss_op_standard(**params)
    else:
      api_name = "ctc_loss_" + str(uuid.uuid4())
      ctc_loss_op_standard = _generate_defun_backend(api_name, _CPU_DEVICE_NAME,
                                                     _ctc_loss_op_standard)
      ctc_loss_op_cudnn = _generate_defun_backend(api_name, _GPU_DEVICE_NAME,
                                                  _ctc_loss_op_cudnn)
      res = ctc_loss_op_standard(**params)
      concrete_func = ctc_loss_op_cudnn.get_concrete_function(**params)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
    return res
  if blank_index is None:
    blank_index = 0
  return ctc_loss_dense(
      labels=labels,
      logits=logits,
      label_length=label_length,
      logit_length=logit_length,
      logits_time_major=logits_time_major,
      unique=unique,
      blank_index=blank_index,
      name=name)
