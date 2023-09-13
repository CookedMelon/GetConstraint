"/home/cc/Workspace/tfconstraint/keras/applications/resnet.py"
@keras_export(
    "keras.applications.resnet50.preprocess_input",
    "keras.applications.resnet.preprocess_input",
)
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )
