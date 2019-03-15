import numpy as np
import keras


class Anchor:
    """ The Class that define how anchors are generated.

    :param sizes : List of sizes to use. Each size corresponds to one feature level.
    :param strides : List of strides to use. Each stride correspond to one feature level.
    :param ratios : List of ratios to use per location in a feature map.
    :param scales  : List of scales to use per location in a feature map.
    :param num_samples : Number of anchors to calculate loss
    :param negative_overlap : IoU overlap for negative anchors
                              (all anchors with overlap < negative_overlap are negative).
    :param positive_overlap : IoU overlap for positive anchors
                              (all anchors with overlap > positive_overlap are positive).
    """

    def __init__(self,
                 sizes=[20],
                 strides=[8],
                 ratios=[1, np.sqrt(2), 1 / 2 * np.sqrt(2)],
                 scales=[1, 1.5, 2, 2.5, 3],
                 num_samples=64,
                 negative_overlap=0.3,
                 positive_overlap=0.6):
        self.sizes = np.array(sizes)
        self.strides = np.array(strides)
        self.ratios = np.array(ratios)
        self.scales = np.array(scales)
        self.num_samples = num_samples
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.validate_size_and_stride()
        self.num_pyramids = len(self.sizes)
        self.num_anchors = len(self.ratios) * len(self.scales)

    def validate_size_and_stride(self):
        if len(self.sizes) != len(self.strides):
            raise ValueError("sizes의 갯수와 strides의 갯수는 일치해야 합니다.")

    def points_to_deltas(self, batch_points, images):
        # construct an anchor object
        max_shape = tuple(
            max(image.shape[x] for image in images) for x in range(2))
        output_shapes = self.guess_feature_map_size(max_shape)
        anchors = self.anchors_for_shape(output_shapes)

        # construct an delta anchor object
        batch_centers = []
        for points in batch_points:
            center_x = points[:, :2].sum(axis=1) / 2
            center_y = points[:, 2:].sum(axis=1) / 2
            width = points[:, 1] - points[:, 0]
            height = points[:, 3] - points[:, 2]

            centers = np.stack([center_x, center_y, width, height], axis=-1)
            batch_centers.append(centers)

        deltas = self.assign_delta(anchors, batch_centers)

        return deltas

    def guess_anchors(self, images):
        # construct an anchor object
        max_shape = tuple(
            max(image.shape[x] for image in images) for x in range(2))
        output_shapes = self.guess_feature_map_size(max_shape)
        anchors = self.anchors_for_shape(output_shapes)
        return anchors

    def guess_feature_map_size(self, input_shape):
        """Guess output shapes based on strides.

        :param input_shape: Shape of the input image.

        :return: A list of image shapes at each stride
        """

        image_shape = np.array(input_shape[:2])
        image_shapes = np.array([np.ceil(image_shape / stride)
                                 for stride in self.strides], np.int32)
        return image_shapes

    def anchors_for_shape(self, output_shapes):
        """ Generators anchors for a given shape.

        :param output_shapes : List of the shape of the output feature maps from FPN network

        :return:
            np.array of shape (N, 4) containing the (c_x, c_y, width, height) coordinates for the anchors.
        """

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4))
        for idx in range(self.num_pyramids):
            seed_anchor = self.generate_seed_anchor(idx)
            anchors = self.roll_anchor(
                output_shapes[idx],
                self.strides[idx],
                seed_anchor)
            all_anchors = np.append(all_anchors, anchors, axis=0)

        return all_anchors

    def generate_seed_anchor(self, output_index):
        """Generate seed anchor (reference) by enumerating aspect ratios X scales w.r.t. a reference window.

        :param output_index : 몇번째 출력층에 대한 seed Anchor인지 정의
        """
        base_size = self.sizes[output_index]

        _anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                anchor = np.array([base_size, base_size]) * scale
                anchor = anchor * np.array((ratio, 1 / ratio))
                _anchors.append(anchor)
        seed_anchor = np.array(_anchors)

        return np.concatenate([np.zeros_like(seed_anchor),
                               seed_anchor], axis=1)

    @staticmethod
    def roll_anchor(
            shape,
            stride,
            seed_anchor
    ):
        """ Produce anchors by rolling seed anchor based on shape of the map and stride size.

        :param shape  : Shape to shift the anchors over.
        :param stride : Stride to shift the anchors with over the shape.
        :param seed_anchor : Seed anchor to apply at each location.
        """
        num_value = seed_anchor.shape[1]  # 3 : (c_x,c_y,width,height)
        roll_shape = (
            shape[0],
            shape[1],
            seed_anchor.shape[0],
            seed_anchor.shape[1])
        # (num_batch, height, width, num_anchor, num_value)
        rolled_anchors = np.zeros(roll_shape)

        # C_X, C_Y 정보
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride
        shifts = np.stack(np.meshgrid(shift_x, shift_y), axis=-1)
        # (height, width, num_anchor, :2)에 정보를 넣음
        rolled_anchors[:, :, :, :2] = shifts[:, :, np.newaxis, :]

        rolled_anchors[:,
                       :,
                       :,
                       2:] = seed_anchor[np.newaxis,
                                         np.newaxis,
                                         :,
                                         2:]
        return rolled_anchors.reshape(-1, num_value)

    def assign_delta(self,
                     anchors,
                     batch_centers):
        """ Generate delta anchors for bbox detection.

        :param anchors: np.array of annotations of shape (N, 4)
                        for (c_x, c_y, w, h)
        :param batch_centers: ground truth

        :returns:
        delta_regs: batch that contains bounding-box regression targets
                    for an image & anchor states
                    (np.array of shape (batch_size, N, 4 + 1),
                    where N is the number of anchors for an image,
                    the first 4 columns define regression targets
                    for (t_x, t_y, t_w, t_h) and the last column
                    defines anchor states
                    (-1 for ignore, 0 for bg, 1 for fg).
        anchor_clfs: batch that contains labels & anchor states
                     (np.array of shape (batch_size, N, num_classes + 1),
                     where N is the number of anchors for an image
                     and the last column defines the anchor state
                     (-1 for ignore, 0 for bg, 1 for fg).
        """
        batch_size = len(batch_centers)

        delta_regs = np.zeros(
            (batch_size,
             anchors.shape[0],
             anchors.shape[1] + 1),
            dtype=keras.backend.floatx())
        anchor_clfs = np.zeros(
            (batch_size,
             anchors.shape[0],
             2 + 1),
            dtype=keras.backend.floatx())

        # compute labels and regression targets
        for index, prior_bbox in enumerate(batch_centers):
            # obtain indices of gt annotations with the greatest overlap
            positive_inds, ignore_inds, class_indices = \
                self.split_anchor_by_overlap(anchors, prior_bbox)

            anchor_clfs[index, ignore_inds, -1] = -1
            anchor_clfs[index, positive_inds, -1] = 1
            anchor_clfs[index, positive_inds, 1] = 1

            delta_regs[index, ignore_inds, -1] = -1
            delta_regs[index, positive_inds, -1] = 1

            delta_regs[index, positive_inds, :-1] = \
                self.calculate_delta(anchors[positive_inds],
                                     prior_bbox[class_indices, :])

        return anchor_clfs, delta_regs

    def split_anchor_by_overlap(self,
                                anchor_bbox,
                                prior_bbox
                                ):
        """ split indices of gt annotations with the greatest overlap.

        :param anchor_bbox: np.array of anchors of shape (N, 4) for (c_x, c_y, width, height).
        :param prior_bbox: np.array of prior of shape (N, 4) for (c_x, c_y, width, height).

        :returns:
            positive_masks: indices of positive anchors
            ignore_masks: indices of ignored anchors
            class_indicies: indices of the class ordered by positive anchor
        """
        overlaps = self.compute_iou(anchor_bbox, prior_bbox)
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(
            overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_masks = (max_overlaps >= self.positive_overlap)
        max_overlaps_inds = np.argmax(overlaps, axis=0)
        positive_masks[max_overlaps_inds] = True

        negative_masks = (max_overlaps <= self.negative_overlap)

        positive_masks, negative_masks = \
            self.subsample_anchors(positive_masks, negative_masks)

        ignore_masks = ~(positive_masks + negative_masks)

        class_indices = argmax_overlaps_inds[positive_masks]

        return positive_masks, ignore_masks, class_indices

    def compute_iou(self, anchor_bbox, prior_bbox):
        """

        IoU used for training

        :param anchor_bbox: np.array of anchors of shape (N, 4) for (c_x, c_y, width, height).
        :param prior_bbox: np.array of prior of shape (M, 4) for (c_x, c_y, width, height).

        :return: overlaps
            np.array of iou of shape (M, N)
            N - the number of anchors
            M - the number of priors
        """
        area_sums = (
            (prior_bbox[:, 2] * prior_bbox[:, 3])[np.newaxis].T
            + (anchor_bbox[:, 2] * anchor_bbox[:, 3])[np.newaxis])

        area_intersections = np.zeros_like(area_sums)

        half_width = (anchor_bbox[:, 2] / 2)
        half_height = (anchor_bbox[:, 3] / 2)

        ac_min_xs = anchor_bbox[:, 0] - half_width
        ac_min_ys = anchor_bbox[:, 1] - half_height
        ac_max_xs = anchor_bbox[:, 0] + half_width
        ac_max_ys = anchor_bbox[:, 1] + half_height

        for idx, true_center in enumerate(prior_bbox):
            c_x, c_y, w, h = true_center
            gt_min_xs = np.ones_like(ac_min_xs) * (c_x - w / 2)
            gt_min_ys = np.ones_like(ac_min_ys) * (c_y - h / 2)
            gt_max_xs = np.ones_like(ac_max_xs) * (c_x + w / 2)
            gt_max_ys = np.ones_like(ac_max_ys) * (c_y + h / 2)

            iw = (np.min([gt_max_xs, ac_max_xs], axis=0)
                  - np.max([gt_min_xs, ac_min_xs], axis=0))
            iw[iw < 0] = 0
            ih = (np.min([gt_max_ys, ac_max_ys], axis=0)
                  - np.max([gt_min_ys, ac_min_ys], axis=0))
            ih[ih < 0] = 0

            area_intersections[idx] = iw * ih

        iou = (area_intersections / (area_sums - area_intersections))

        return iou.T

    def calculate_delta(self, anchors, gt_boxes):
        """Compute bounding-box regression targets for an image."""
        delta_cx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
        delta_cy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
        delta_w = np.log(gt_boxes[:, 2] / anchors[:, 2])
        delta_h = np.log(gt_boxes[:, 3] / anchors[:, 3])

        targets = np.stack((delta_cx, delta_cy, delta_w, delta_h))
        targets = targets.T

        return targets

    def subsample_anchors(self, positive_masks, negative_masks):
        positive_indices = np.argwhere(positive_masks).ravel()
        negative_indices = np.argwhere(negative_masks).ravel()

        # Subsampling Positive and Negative indicies
        num_positives = np.clip(
            len(positive_indices), 0, self.num_samples // 2)
        positive_indices = np.random.choice(positive_indices,
                                            num_positives, replace=False)

        num_negatives = self.num_samples - num_positives
        negative_indices = np.random.choice(negative_indices,
                                            num_negatives, replace=False)

        positive_masks = np.zeros_like(positive_masks, np.bool)
        positive_masks[positive_indices] = True

        negative_masks = np.zeros_like(positive_masks, np.bool)
        negative_masks[negative_indices] = True

        return positive_masks, negative_masks
