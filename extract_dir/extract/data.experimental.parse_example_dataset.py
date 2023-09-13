@tf_export("data.experimental.parse_example_dataset")
@deprecation.deprecated(
    None, "Use `tf.data.Dataset.map(tf.io.parse_example(...))` instead.")
