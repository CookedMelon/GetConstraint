@tf_export("io.decode_raw", v1=[])
@dispatch.add_dispatch_support
def decode_raw(input_bytes,
               out_type,
               little_endian=True,
               fixed_length=None,
               name=None):
  r"""Convert raw bytes from input tensor into numeric tensors.
  Every component of the input tensor is interpreted as a sequence of bytes.
  These bytes are then decoded as numbers in the format specified by `out_type`.
  >>> tf.io.decode_raw(tf.constant("1"), tf.uint8)
  <tf.Tensor: shape=(1,), dtype=uint8, numpy=array([49], dtype=uint8)>
  >>> tf.io.decode_raw(tf.constant("1,2"), tf.uint8)
  <tf.Tensor: shape=(3,), dtype=uint8, numpy=array([49, 44, 50], dtype=uint8)>
  Note that the rank of the output tensor is always one more than the input one:
  >>> tf.io.decode_raw(tf.constant(["1","2"]), tf.uint8).shape
  TensorShape([2, 1])
  >>> tf.io.decode_raw(tf.constant([["1"],["2"]]), tf.uint8).shape
  TensorShape([2, 1, 1])
  This is because each byte in the input is converted to a new value on the
  output (if output type is `uint8` or `int8`, otherwise chunks of inputs get
  coverted to a new value):
  >>> tf.io.decode_raw(tf.constant("123"), tf.uint8)
  <tf.Tensor: shape=(3,), dtype=uint8, numpy=array([49, 50, 51], dtype=uint8)>
  >>> tf.io.decode_raw(tf.constant("1234"), tf.uint8)
  <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([49, 50, 51, 52], ...
  >>> # chuncked output
  >>> tf.io.decode_raw(tf.constant("12"), tf.uint16)
  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([12849], dtype=uint16)>
  >>> tf.io.decode_raw(tf.constant("1234"), tf.uint16)
  <tf.Tensor: shape=(2,), dtype=uint16, numpy=array([12849, 13363], ...
  >>> # int64 output
  >>> tf.io.decode_raw(tf.constant("12345678"), tf.int64)
  <tf.Tensor: ... numpy=array([4050765991979987505])>
  >>> tf.io.decode_raw(tf.constant("1234567887654321"), tf.int64)
  <tf.Tensor: ... numpy=array([4050765991979987505, 3544952156018063160])>
  The operation allows specifying endianness via the `little_endian` parameter.
  >>> tf.io.decode_raw(tf.constant("\x0a\x0b"), tf.int16)
  <tf.Tensor: shape=(1,), dtype=int16, numpy=array([2826], dtype=int16)>
  >>> hex(2826)
  '0xb0a'
  >>> tf.io.decode_raw(tf.constant("\x0a\x0b"), tf.int16, little_endian=False)
  <tf.Tensor: shape=(1,), dtype=int16, numpy=array([2571], dtype=int16)>
  >>> hex(2571)
  '0xa0b'
  If the elements of `input_bytes` are of different length, you must specify
  `fixed_length`:
  >>> tf.io.decode_raw(tf.constant([["1"],["23"]]), tf.uint8, fixed_length=4)
  <tf.Tensor: shape=(2, 1, 4), dtype=uint8, numpy=
  array([[[49,  0,  0,  0]],
         [[50, 51,  0,  0]]], dtype=uint8)>
  If the `fixed_length` value is larger that the length of the `out_type` dtype,
  multiple values are generated:
  >>> tf.io.decode_raw(tf.constant(["1212"]), tf.uint16, fixed_length=4)
  <tf.Tensor: shape=(1, 2), dtype=uint16, numpy=array([[12849, 12849]], ...
  If the input value is larger than `fixed_length`, it is truncated:
  >>> x=''.join([chr(1), chr(2), chr(3), chr(4)])
  >>> tf.io.decode_raw(x, tf.uint16, fixed_length=2)
  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([513], dtype=uint16)>
  >>> hex(513)
  '0x201'
  If `little_endian` and `fixed_length` are specified, truncation to the fixed
  length occurs before endianness conversion:
  >>> x=''.join([chr(1), chr(2), chr(3), chr(4)])
  >>> tf.io.decode_raw(x, tf.uint16, fixed_length=2, little_endian=False)
  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([258], dtype=uint16)>
  >>> hex(258)
  '0x102'
  If input values all have the same length, then specifying `fixed_length`
  equal to the size of the strings should not change output:
  >>> x = ["12345678", "87654321"]
  >>> tf.io.decode_raw(x, tf.int16)
  <tf.Tensor: shape=(2, 4), dtype=int16, numpy=
  array([[12849, 13363, 13877, 14391],
         [14136, 13622, 13108, 12594]], dtype=int16)>
  >>> tf.io.decode_raw(x, tf.int16, fixed_length=len(x[0]))
  <tf.Tensor: shape=(2, 4), dtype=int16, numpy=
  array([[12849, 13363, 13877, 14391],
         [14136, 13622, 13108, 12594]], dtype=int16)>
  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
      Currently, this must be a tensor of strings (bytes), although semantically
      the operation should support any input.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    fixed_length:
      If set, the first `fixed_length` bytes of each element will be converted.
      Data will be zero-padded or truncated to the specified length.
      `fixed_length` must be a multiple of the size of `out_type`.
      `fixed_length` must be specified if the elements of `input_bytes` are of
      variable length.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` object storing the decoded bytes.
  """
  if fixed_length is not None:
    return gen_parsing_ops.decode_padded_raw(
        input_bytes,
        fixed_length=fixed_length,
        out_type=out_type,
        little_endian=little_endian,
        name=name)
  else:
    return gen_parsing_ops.decode_raw(
        input_bytes, out_type, little_endian=little_endian, name=name)
