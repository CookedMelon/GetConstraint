@tf_export("tpu.experimental.embedding.QuantizationConfig")
class QuantizationConfig:
  """Settings for simulated quantization of the tpu embedding table.
  When simulated quantization is enabled, the results of the embedding lookup
  are clipped and quantized according to the settings here before the combiner
  is applied.
  For example, to quantize `input` the following is done:
  ```python
  if input < lower
    input = lower
  if input > upper
    input = upper
  quantum = (upper - lower) / (num_buckets - 1)
  input = math.floor((input - lower) / quantum + 0.5) * quantium + lower
  ```
  See tensorflow/core/protobuf/tpu/optimization_parameters.proto for more
  details.
  NOTE: This does not change the storage type of the embedding table, that will
  continue to be float32 as will the saved variable in the checkpoint. You will
  have to manually quantize the variable (typically with the same algorithm and
  settings as above) manually.
  """
  def __init__(self, num_buckets: int, lower: float, upper: float):
    """Simulated quantizaiton configuration.
    Args:
      num_buckets: The number of quantization buckets, must be atleast 2.
      lower: The lower bound for the quantization range.
      upper: The upper bound for the quantization range.
    Returns:
      `QuantizationConfig`.
    Raises:
      ValueError: if `num_buckets` is less than 2.
    """
    if num_buckets < 2:
      raise ValueError(f"num_buckets is {num_buckets}, must be at least 2 for "
                       f"simulated quantization.")
    self.num_buckets = num_buckets
    self.lower = lower
    self.upper = upper
  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    parameters.simulated_quantization.enabled = True
    parameters.simulated_quantization.num_buckets = self.num_buckets
    parameters.simulated_quantization.clipping_limits.lower.value = self.lower
    parameters.simulated_quantization.clipping_limits.upper.value = self.upper
  def __repr__(self):
    return ("QuantizationConfig(num_buckets={num_buckets!r}, lower={lower!r}, "
            "upper={upper!r})".format(
                num_buckets=self.num_buckets,
                lower=self.lower,
                upper=self.upper))
