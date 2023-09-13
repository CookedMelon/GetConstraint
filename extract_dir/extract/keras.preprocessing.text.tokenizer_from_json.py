@keras_export("keras.preprocessing.text.tokenizer_from_json")
def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration and returns a tokenizer instance.
    Deprecated: `tf.keras.preprocessing.text.Tokenizer` does not operate on
    tensors and is not recommended for new code. Prefer
    `tf.keras.layers.TextVectorization` which provides equivalent functionality
    through a layer which accepts `tf.Tensor` input. See the
    [text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
    for an overview of the layer and text handling in tensorflow.
    Args:
        json_string: JSON string encoding a tokenizer configuration.
    Returns:
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get("config")
    word_counts = json.loads(config.pop("word_counts"))
    word_docs = json.loads(config.pop("word_docs"))
    index_docs = json.loads(config.pop("index_docs"))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop("index_word"))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop("word_index"))
    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word
    return tokenizer
