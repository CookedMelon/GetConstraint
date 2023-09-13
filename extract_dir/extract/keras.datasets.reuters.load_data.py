@keras_export("keras.datasets.reuters.load_data")
def load_data(
    path="reuters.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs,
