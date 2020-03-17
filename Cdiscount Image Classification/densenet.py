# Adapted from https://github.com/flyyufelix/DenseNet-Keras
import os
from urllib.request import urlretrieve

import keras.backend as K
import numpy as np
from keras import initializers as initializations
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.engine import InputSpec, Layer
from keras.layers import Conv2D, Input, ZeroPadding2D, concatenate
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from scipy.misc import imread, imresize

# Sanity check
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.expanduser("~"), ".keras/models", "densenet121_weights_tf.h5"
)
assert os.path.isfile(
    DEFAULT_WEIGHTS_PATH
), "Download the pre-trained weights from https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc to {}!".format(
    DEFAULT_WEIGHTS_PATH
)
assert K.backend() == "tensorflow" and K.image_data_format() == "channels_last"


class Scale(Layer):
    """Custom Layer for DenseNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where "gamma" and "beta" are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            "[(input_shape,), (input_shape,)]"
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a "weights" argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a "weights" argument.
    """

    def __init__(
        self,
        weights=None,
        axis=-1,
        momentum=0.9,
        beta_init="zero",
        gamma_init="one",
        **kwargs
    ):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(
            self.gamma_init(shape), name="{}_gamma".format(self.name)
        )
        self.beta = K.variable(self.beta_init(shape), name="{}_beta".format(self.name))
        # self.gamma = self.gamma_init(shape, name="{}_gamma".format(self.name))
        # self.beta = self.beta_init(shape, name="{}_beta".format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):  # @UnusedVariable
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(
            self.beta, broadcast_shape
        )
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv_block(
    x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4
):  # @UnusedVariable
    """Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
    # Arguments
        x: input tensor
        stage: index for dense block
        branch: layer index within each dense block
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = "conv" + str(stage) + "_" + str(branch)
    relu_name_base = "relu" + str(stage) + "_" + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name=conv_name_base + "_x1_bn"
    )(x)
    x = Scale(axis=concat_axis, name=conv_name_base + "_x1_scale")(x)
    x = Activation("relu", name=relu_name_base + "_x1")(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base + "_x1", use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name=conv_name_base + "_x2_bn"
    )(x)
    x = Scale(axis=concat_axis, name=conv_name_base + "_x2_scale")(x)
    x = Activation("relu", name=relu_name_base + "_x2")(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + "_x2_zeropadding")(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base + "_x2", use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(
    x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1e-4
):  # @UnusedVariable
    """Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
    # Arguments
        x: input tensor
        stage: index for dense block
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = "conv" + str(stage) + "_blk"
    relu_name_base = "relu" + str(stage) + "_blk"
    pool_name_base = "pool" + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + "_bn")(
        x
    )
    x = Scale(axis=concat_axis, name=conv_name_base + "_scale")(x)
    x = Activation("relu", name=relu_name_base)(x)
    x = Conv2D(
        int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False
    )(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(
    x,
    stage,
    nb_layers,
    nb_filter,
    growth_rate,
    dropout_rate=None,
    weight_decay=1e-4,
    grow_nb_filters=True,
):
    """Build a dense_block where the output of each conv_block is fed to subsequent ones
    # Arguments
        x: input tensor
        stage: index for dense block
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
    """
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(
            concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay
        )
        concat_feat = concatenate(
            [concat_feat, x],
            axis=concat_axis,
            name="concat_" + str(stage) + "_" + str(branch),
        )

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def DenseNet(
    nb_dense_block=4,
    growth_rate=32,
    nb_filter=64,
    reduction=0.5,
    dropout_rate=0.0,
    weight_decay=1e-4,
    classes=1000,
    weights_path=DEFAULT_WEIGHTS_PATH,
    last_trainable_layer_name="pool4",
):
    """Instantiate the DenseNet 121 architecture,
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
        last_trainable_layer_name: name of the last trainable layer
    # Returns
        A Keras model instance.
    """
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == "tf":
        concat_axis = 3
        img_input = Input(shape=(None, None, 3), name="data")
    else:
        concat_axis = 1
        img_input = Input(shape=(3, None, None), name="data")

    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name="conv1", use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name="conv1_bn")(x)
    x = Scale(axis=concat_axis, name="conv1_scale")(x)
    x = Activation("relu", name="relu1")(x)
    x = ZeroPadding2D((1, 1), name="pool1_zeropadding")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(
            x,
            stage,
            nb_layers[block_idx],
            nb_filter,
            growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )

        # Add transition_block
        x = transition_block(
            x,
            stage,
            nb_filter,
            compression=compression,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(
        x,
        final_stage,
        nb_layers[-1],
        nb_filter,
        growth_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    x = BatchNormalization(
        epsilon=eps, axis=concat_axis, name="conv" + str(final_stage) + "_blk_bn"
    )(x)
    x = Scale(axis=concat_axis, name="conv" + str(final_stage) + "_blk_scale")(x)
    x = Activation("relu", name="relu" + str(final_stage) + "_blk")(x)
    x = GlobalAveragePooling2D(name="pool" + str(final_stage))(x)

    x = Dense(classes, name="fc6")(x)
    x = Activation("softmax", name="prob")(x)

    model = Model(img_input, x, name="densenet")

    if weights_path is not None:
        print("Loading weights from {} ...".format(weights_path))
        model.load_weights(weights_path, by_name=True)

    if last_trainable_layer_name is not None:
        print("Freezing all layers until {} ...".format(last_trainable_layer_name))
        for layer in model.layers:
            layer.trainable = False
            if layer.name == last_trainable_layer_name:
                break

    return model


def preprocessing_function(sample):
    return preprocess_input(np.array([sample], dtype=np.float32))[0] * 0.017


def run(image_URL="http://dreamicus.com/data/lake/lake-04.jpg"):
    print("Loading DenseNet ...")
    model = DenseNet()
    optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    plot_model(
        model,
        to_file=os.path.join("/tmp", "model.png"),
        show_shapes=True,
        show_layer_names=True,
    )

    image_path = os.path.join("/tmp", image_URL.split("/")[-1])
    if not os.path.isfile(image_path):
        print("Downloading the image from {} ...".format(image_URL))
        urlretrieve(image_URL, image_path)
    assert os.path.isfile(image_path)

    print("Generating prediction ...")
    image_content = imread(image_path)
    image_content = imresize(image_content, (224, 224))
    image_content = preprocessing_function(image_content)
    prediction_array = model.predict(np.array([image_content]))
    print("Prediction:", decode_predictions(prediction_array))

    print("All done!")


if __name__ == "__main__":
    run()
