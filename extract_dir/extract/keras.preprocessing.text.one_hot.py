@keras_export("keras.preprocessing.text.one_hot")
def one_hot(
    input_text,
    n,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    analyzer=None,
