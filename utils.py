import itertools
import math
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np


def generate_default_boxes():

    default_boxes = []
    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
    fm_sizes = [38, 19, 10, 5, 3, 1]
    ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    image_size = 300

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes


def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2

    hw = tf.clip_by_value(bot_right - top_left, 0.0, 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    """ Compute regression and classification targets
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        gt_boxes: tensor (num_gt, 4)
                  of format (xmin, ymin, xmax, ymax)
        gt_labels: tensor (num_gt,)
    Returns:
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    """
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    transformed_default_boxes = transform_center_to_corner(default_boxes)
    iou = compute_iou(transformed_default_boxes, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    # Normal way: use a for loop
    # for gt_idx, default_idx in enumerate(best_default_idx):
    #     best_gt_idx = tf.tensor_scatter_nd_update(
    #         best_gt_idx,
    #         tf.expand_dims([default_idx], 1),
    #         [gt_idx])

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs


def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = transform_corner_to_center(boxes)

    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis=-1)

    return locs


def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)

    boxes = transform_center_to_corner(locs)

    return boxes


def transform_corner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep
    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)


class ImageVisualizer(object):
    """ Class for visualizing image
    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir
        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')


def generate_patch(boxes, threshold):
    """ Function to generate a random patch within the image
        If the patch overlaps any gt boxes at above the threshold,
        then the patch is picked, otherwise generate another patch
    Args:
        boxes: box tensor (num_boxes, 4)
        threshold: iou threshold to decide whether to choose the patch
    Returns:
        patch: the picked patch
        ious: an array to store IOUs of the patch and all gt boxes
    """
    while True:
        patch_w = random.uniform(0.1, 1)
        scale = random.uniform(0.5, 2)
        patch_h = patch_w * scale
        patch_xmin = random.uniform(0, 1 - patch_w)
        patch_ymin = random.uniform(0, 1 - patch_h)
        patch_xmax = patch_xmin + patch_w
        patch_ymax = patch_ymin + patch_h
        patch = np.array(
            [[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
            dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)
        ious = compute_iou(tf.constant(patch), boxes)
        if tf.math.reduce_any(ious >= threshold):
            break

    return patch[0], ious[0]


def random_patching(img, boxes, labels):
    """ Function to apply random patching
        Firstly, a patch is randomly picked
        Then only gt boxes of which IOU with the patch is above a threshold
        and has center point lies within the patch will be selected
    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    Returns:
        img: the cropped PIL Image
        boxes: selected gt boxes tensor (new_num_boxes, 4)
        labels: selected gt labels tensor (new_num_boxes,)
    """
    threshold = np.random.choice(np.linspace(0.1, 0.7, 4))

    patch, ious = generate_patch(boxes, threshold)

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep_idx = (
        (ious > 0.3) &
        (box_centers[:, 0] > patch[0]) &
        (box_centers[:, 1] > patch[1]) &
        (box_centers[:, 0] < patch[2]) &
        (box_centers[:, 1] < patch[3])
    )

    if not tf.math.reduce_any(keep_idx):
        return img, boxes, labels

    img = img.crop(patch)

    boxes = boxes[keep_idx]
    patch_w = patch[2] - patch[0]
    patch_h = patch[3] - patch[1]
    boxes = tf.stack([
        (boxes[:, 0] - patch[0]) / patch_w,
        (boxes[:, 1] - patch[1]) / patch_h,
        (boxes[:, 2] - patch[0]) / patch_w,
        (boxes[:, 3] - patch[1]) / patch_h], axis=1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    labels = labels[keep_idx]

    return img, boxes, labels


def horizontal_flip(img, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly
    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    Returns:
        img: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)
    
    return img, boxes, labels
    
def vertical_flip(img, boxes, labels):
   """ Function to vertically flip the image
       The gt boxes will be need to be modified accordingly
   Args:
       img: the original PIL Image
       boxes: gt boxes tensor (num_boxes, 4)
       labels: gt labels tensor (num_boxes,)
   Returns:
       img: the horizontally flipped PIL Image
       boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
       labels: gt labels tensor (num_boxes,)
   """
   img = img.transpose(Image.FLIP_TOP_BOTTOM)
   boxes = tf.stack([
       boxes[:, 0],
       1 - boxes[:, 3],
       boxes[:, 2],
       1 - boxes[:, 1]], axis=1)

   return img, boxes, labels

class ImageVisualizer(object):
    """ Class for visualizing image
    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir
        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor=(0., 1., 0.),
                facecolor="none"))
            # plt.text(
            #     box[0],
            #     box[1],
            #     s=cls_name,
            #     color="white",
            #     verticalalignment="top",
            #     bbox={"color": (0., 1., 0.), "pad": 0},
            # )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')
