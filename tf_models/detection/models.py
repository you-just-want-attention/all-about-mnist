from keras.layers import Input, Conv2D, Reshape, Softmax
from keras.models import Model


def tiny_rpn_network(input_shape, num_anchors):
    img_input, feature_maps = tiny_vgg_base_network(input_shape)
    rpn_probs, rpn_bbox = region_proposal_network(feature_maps, num_anchors)
    return Model(img_input, [rpn_probs, rpn_bbox])


def tiny_vgg_base_network(input_shape):
    """Implements the base network for generating feature maps

    VGG-Style Network but remove max-pooling layer

    returns:
        classification tensor : [batch size, classes]

    """
    img_input = Input(input_shape)

    block_name = "block1"
    x = Conv2D(8, (7, 7),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(img_input)
    x = Conv2D(8, (7, 7), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    block_name = "block2"
    x = Conv2D(16, (5, 5),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(16, (5, 5),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)
    x = Conv2D(16, (5, 5), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv3")(x)

    block_name = "block3"
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)
    x = Conv2D(32, (3, 3), strides=2,
               activation='relu',
               padding="same",
               name=f"{block_name}_conv3")(x)

    block_name = "block4"
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv1")(x)
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding="same",
               name=f"{block_name}_conv2")(x)

    feature_map = x
    return img_input, feature_map


def region_proposal_network(feature_map, num_anchors):
    shared = Conv2D(128, (3, 3), padding='same',
                    activation='relu',
                    name='rpn_conv_shared')(feature_map)

    # Anchor Score, [batch, height, width, anchors per location * 2]
    scores = Conv2D(2 * num_anchors, (1, 1),
                    activation='linear',
                    name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = Reshape((-1, 2))(scores)

    # Softmax on last dimension of BG/FG
    rpn_probs = Softmax(name='rpn_probs')(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    deltas = Conv2D(4 * num_anchors, (1, 1),
                    activation='linear',
                    name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = Reshape((-1, 4),name='rpn_bbox')(deltas)

    return [rpn_probs, rpn_bbox]