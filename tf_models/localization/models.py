from keras.layers import Input, Conv2D, Layer, Flatten, Dense
from keras.models import Model
import tensorflow as tf


def tiny_localization_network(input_shape, classes):
    img_input, feature_maps = tiny_vgg_base_network(input_shape)
    loc_output = localization_subnetwork(feature_maps)
    clf_output = classification_subnetwork(feature_maps, classes)
    return Model(img_input, [loc_output, clf_output])


def tiny_vgg_base_network(input_shape):
    """Implements the base network for generating feature maps

    VGG-Style Network but remove max-pooling layer

    returns:
        classification tensor : [batch size, classes]

    """
    img_input = Input(input_shape)

    block_name = "block1"
    x = Conv2D(8, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(img_input)
    x = Conv2D(8, (3, 3), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    block_name = "block2"
    x = Conv2D(16, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(16, (3, 3), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    block_name = "block3"
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(32, (3, 3), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    block_name = "block4"
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(64, (3, 3), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    # We don't need to care the size of input,
    # ResizeBilinear layer change the shape to (3,3)
    # regardless of the size of the feature map
    x = ResizeBilinear((3, 3))(x)
    x = Flatten()(x)

    return img_input, x


def classification_subnetwork(input_tensor, classes):
    """Implements the sub network for object classification

    returns:
        classification tensor : [batch size, classes]

    """
    clf_x = Dense(32,
                  activation='relu',
                  name="classification_fc1")(input_tensor)
    clf_x = Dense(32,
                  activation='relu',
                  name="classification_fc2")(clf_x)
    clf_out = Dense(classes,
                    activation='softmax',
                    name="classification_output")(clf_x)
    return clf_out


def localization_subnetwork(input_tensor):
    """Implements the sub network for object localization

    returns:
        regression tensor : [batch size, center_x,center_y,width,height]

    """
    loc_x = Dense(32,
                  activation='relu',
                  name="localization_fc1")(input_tensor)
    loc_x = Dense(32,
                  activation='relu',
                  name="localization_fc2")(loc_x)
    loc_out = Dense(4,
                    activation='linear',
                    name="localization_output")(loc_x)
    return loc_out


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


class ResizeBilinear(Layer):
    """ Resizes the feature map to a specific size by bilinear Interpolation

    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ResizeBilinear, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return tf.image.resize_bilinear(x, self.output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_dim, input_shape[-1])