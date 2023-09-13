@keras_export("keras.applications.mobilenet_v3.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)
