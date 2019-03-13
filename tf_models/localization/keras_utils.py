import keras
import tensorflow as tf
import numpy as np
import cv2
import keras.backend as K


class TensorBoardImage(keras.callbacks.Callback):
    """ Implements the callback of Tensorboard summary writer to draw image
    """
    def __init__(self, images, true_points, true_labels, log_path="./logs"):
        super().__init__()
        self.images = images
        self.true_points = true_points
        self.true_labels = true_labels
        self.num_cases = len(self.images)

        self.log_path = log_path
        self.build_summary_graph()
        self.writer = tf.summary.FileWriter(self.log_path)

    def build_summary_graph(self):
        """ Append the Graph nodes for summarizing image

        텐서플로우의 그래프에 새로운 summary operation을 추가한다.
        이를 통해, 우리는 주어진 placeholder로 이미지를 넣으면,
        tensorboard에서 시각화할 수 있는 이미지(<tf.summary.image>)로 인코딩된다.
        """
        for idx in range(self.num_cases):
            tf_input = tf.placeholder(tf.uint8,
                                      shape=(None, None, None, 3),
                                      name="{}th_image_input".format(idx))
            tf.summary.image("{}th_image".format(idx), tf_input)

    def on_epoch_end(self, epoch, logs={}):
        """

        :param epoch:
        :param logs:
        :return:
        """
        pred_centers, pred_labels = self.model.predict(self.images)

        height, width = self.images.shape[1:3]
        rescaling = np.array([width, width, height, height])
        pred_points = (center2minmax(pred_centers.reshape(-1, 4))
                       * rescaling).astype(np.int)
        true_points = (center2minmax(self.true_points.reshape(-1,4))
                       * rescaling).astype(np.int)

        for idx, (image, pred_point, pred_label, true_point, true_label) in enumerate(zip(
                self.images, pred_points, pred_labels, true_points, self.true_labels)):
            image = visualize_result(image, pred_point, pred_label, true_point, true_label)
            summary_out = K.get_session().run("{}th_image:0".format(idx), feed_dict={
                "{}th_image_input:0".format(idx): image[np.newaxis]
            })
            self.writer.add_summary(summary_out, epoch)
        return


def visualize_result(image, pred_point, pred_label, true_point, true_label):
    """
    Visualize how the model is looking for the objects
    by comparing correct answers and predictions

    :param image: np.ndarray for image
    :param pred_point: np.ndarray consisting of [batch size, x_min, x_max, y_min, y_max]
    :param pred_label: one-hot vector for classes
    :param true_point: np.ndarray consisting of [batch size, x_min, x_max, y_min, y_max]
    :param true_label: one-hot vector for classes
    :return:
        Image with bounding box and Label
        Red color -> Correct answer
        Green Color -> Prediction
    """
    image = image.copy()

    if not image.dtype == np.uint8:
        image = (image*255).astype(np.uint8)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # visualize bounding box
    image = cv2.rectangle(image,
                          (pred_point[0], pred_point[2]),
                          (pred_point[1], pred_point[3]), (255, 0, 0), 1)
    image = cv2.rectangle(image,
                          (true_point[0], true_point[2]),
                          (true_point[1], true_point[3]), (0, 255, 0), 1)

    # visualize prediction value
    text = "{}".format(pred_label.argmax())
    image = cv2.putText(image, text, (0, 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 0, 0))
    text = "{}".format(true_label.argmax())
    image = cv2.putText(image, text, (image.shape[1] - 20, 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 0))

    return image

"""
Keras Custom Metric Function

Custom Metrics can be passed at the compilation step and 
we should implement it by tensorflow graph. 

````python
    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', mean_pred])
````
"""


def iou_metric(y_true, y_pred):
    """ Implements metric graph for calculating Intersection-over-union

    """
    pred_centers = tf.reshape(y_pred, shape=(-1, 4))
    true_centers = tf.reshape(y_true, shape=(-1, 4))

    def tf_center2minmax(centers):
        center_x = centers[:, 0]
        center_y = centers[:, 1]
        half_width = (tf.exp(centers[:, 2]) - 1) / 2
        half_height = (tf.exp(centers[:, 3]) - 1) / 2
        min_x, max_x = center_x - half_width, center_x + half_width
        min_y, max_y = center_y - half_height, center_y + half_height
        return tf.stack([min_x, max_x, min_y, max_y], axis=-1)

    pred_points = tf_center2minmax(pred_centers)
    true_points = tf_center2minmax(true_centers)

    pred_area = (pred_points[:, 1] - pred_points[:, 0]) * (
            pred_points[:, 3] - pred_points[:, 2])
    true_area = (true_points[:, 1] - true_points[:, 0]) * (
            true_points[:, 3] - true_points[:, 2])

    i_min_x = tf.maximum(pred_points[:, 0], true_points[:, 0])
    i_max_x = tf.minimum(pred_points[:, 1], true_points[:, 1])
    i_min_y = tf.maximum(pred_points[:, 2], true_points[:, 2])
    i_max_y = tf.minimum(pred_points[:, 3], true_points[:, 3])
    i_width = tf.clip_by_value((i_max_x - i_min_x), 0., np.inf)
    i_height = tf.clip_by_value((i_max_y - i_min_y), 0., np.inf)

    intersection_area = i_width * i_height
    union_area = pred_area + true_area - intersection_area
    return tf.reduce_mean(intersection_area / union_area)


def accuracy_metric(y_true, y_pred):
    """ Implements metric graph for calculating accuracy

    """
    y_true = tf.reshape(y_true, shape=[-1,10])
    y_pred = tf.reshape(y_pred, shape=[-1,10])
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    correct = tf.equal(true_labels, pred_labels)
    correct = tf.cast(correct, tf.float32)
    return tf.reduce_mean(correct)


"""
Two Style Cordinates Representation

1. minmax style :
[batch_size, x_min, x_max, y_min, y_max]

x_min : the leftmost coordinates of an object
x_max : the rightmost coordinates of an object
y_min : the uppermost coordinates of an object
y_max : the lowest coordinates of an object

2. center style : 
[batch_size, center_x, center_y, width, height]

center_x : center x coordinates of an object
center_y : center y coordinates of an object
width  : the log scale about the width of object 
height : the log scale about the height of object


minmax2center : minmax style -> center style
center2minmax : center style -> minmax style
"""


def minmax2center(points):
    if points.ndim == 2:
        center_x = points[:, :2].sum(axis=1) / 2
        center_y = points[:, 2:].sum(axis=1) / 2
        # Normalization by log scale
        width = np.log(points[:, 1] - points[:, 0] + 1)
        height = np.log(points[:, 3] - points[:, 2] + 1)
        return np.stack([center_x, center_y, width, height], axis=-1)
    elif points.ndim == 1:
        center_x = (points[0] + points[1]) / 2
        center_y = (points[2] + points[3]) / 2
        # Normalization by log scale
        width = np.log(points[1] - points[0] + 1)
        height = np.log(points[3] - points[2] + 1)
        return np.array([center_x, center_y, width, height])
    else:
        raise ValueError("Available dimensions of points : 1 or 2")


def center2minmax(centers):
    if centers.ndim == 2:
        center_x = centers[:,0]
        center_y = centers[:,1]
        half_width = (np.exp(centers[:,2])-1)/2
        half_height = (np.exp(centers[:,3])-1)/2
        min_x, max_x = center_x - half_width, center_x + half_width
        min_y, max_y = center_y - half_height, center_y + half_height
        return np.stack([min_x, max_x, min_y, max_y],axis=-1)
    elif centers.ndim == 1:
        center_x, center_y = centers[:2]
        half_width = (np.exp(centers[2])-1)/2
        half_height = (np.exp(centers[3])-1)/2
        min_x, max_x = center_x - half_width, center_x + half_width
        min_y, max_y = center_y - half_height, center_y + half_height
        return np.array([min_x, max_x, min_y, max_y])
    else:
        raise ValueError("Available dimensions of centers : 1 or 2")