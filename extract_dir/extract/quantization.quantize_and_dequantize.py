@tf_export("quantization.quantize_and_dequantize")
@dispatch.add_dispatch_support
@deprecation.deprecated(None,
                        "This Op has been deprecated, use" +
                        "`quantize_and_dequantize_v2` instead. To " +
                        "To simulate the V1 the behavior of " +
                        "tf.quantization.quantize_and_dequantize(...) use " +
                        "tf.grad_pass_through(" +
                        "tf.quantization.quantize_and_dequantize_v2)(...).")
