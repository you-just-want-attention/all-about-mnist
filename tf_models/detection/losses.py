import tensorflow as tf
import keras


def classification_loss(y_true, y_pred):
    """
    :param y_true: Tensor from the generator of shape (B, N, 3).
    The last value for each box is the state of the anchor (ignore, negative, positive).
    :param y_pred: Tensor from the network of shape (B, N, 4).
    :return:
        The smooth L1 loss of y_pred w.r.t. y_true
    """
    # separate target and state
    labels = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
    classification = y_pred

    # filter out "ignore" anchors
    indices = tf.where(keras.backend.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    cls_loss = tf.losses.softmax_cross_entropy(labels,classification)

    return keras.backend.mean(cls_loss)


def regression_loss(y_true, y_pred):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5).
            The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true
    """

    regression = y_pred
    # separate target and state
    regression_target = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]

    # filter out "ignore" anchors
    positive_indices = tf.where(keras.backend.equal(anchor_state, 1))
    positive_regression = tf.gather_nd(regression, positive_indices)
    pos_regression_target = tf.gather_nd(regression_target, positive_indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (x)^2                  if |x| < sigma
    #      = sigma * |x| - 0.5 * sigma^2  otherwise
    regression_diff = keras.backend.abs(positive_regression - pos_regression_target)
    l2_loss = 0.5 * keras.backend.pow(regression_diff, 2)
    l1_loss = regression_diff - 0.5 ** 2

    loss = tf.where(
        keras.backend.less(regression_diff, 1),
        l2_loss, l1_loss)

    # compute the normalizer: the number of positive anchors
    normalizer = keras.backend.maximum(1, keras.backend.shape(positive_indices)[0])
    normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

    return keras.backend.sum(loss) / normalizer
