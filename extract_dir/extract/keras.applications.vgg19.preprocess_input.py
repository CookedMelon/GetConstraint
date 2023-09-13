@keras_export("keras.applications.vgg19.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )
