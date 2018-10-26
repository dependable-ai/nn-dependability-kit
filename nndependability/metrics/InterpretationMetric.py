import os
import math
import random
import json
import datetime
import getopt
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from nndependability.metrics import saliency
import visualization

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#class InterpretationPrecisionMetric():
#    """Interpretation precision metric
#    """


#def __init__(self):

def compute_heatmaps_bounding_boxes(original_classes, original_bboxes, stride, img):
    grey = np.array([128,128,128])
    iou_treshold = 0.7

    heatmaps = np.zeros([len(original_classes)]+[(x // stride) + 1 for x in img.shape[:2]])

    [offset_x, offset_y] = [(x % stride) // 2 for x in img.shape[:2]]
    for i in range(heatmaps.shape[1]):
        for j in range(heatmaps.shape[2]):
            x = i * stride + offset_x
            y = j * stride + offset_y
            x1 = max(0, x - grey_box_x // 2)
            x2 = min(img.shape[0], x + grey_box_x // 2) 
            y1 = max(0, y - grey_box_y // 2)
            y2 = min(img.shape[1], y + grey_box_y // 2) 
            old_pix = np.copy(img[x1 : x2, y1 : y2])
            img[x1 : x2, y1 : y2] = grey

            rclasses, rscores, rbboxes, _, _ =  process_image(img)
            for bbox_idx, bbox in enumerate(rbboxes):
                for idx, original_bbox in enumerate(original_bboxes):
                    if iou(bbox, original_bbox) > iou_treshold:
                        #print('found box',iou(bbox, original_bbox), 'class', rclasses[bbox_idx])
                        if rclasses[bbox_idx] == original_classes[idx]:
                            #print('class matches', rscores[bbox_idx])
                            heatmaps[idx,i,j] = max(heatmaps[idx,i,j], rscores[bbox_idx])

            img[x1 : x2, y1 : y2] = old_pix
            print(i*heatmaps.shape[2]+j+1,heatmaps[0].size)
    return heatmaps

def compute_saliency(graph, isess, logits, image_input_tensors, image_input, performance='precise'):
    # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    if performance=='fast':
        gradient_saliency = saliency.GradientSaliency(graph, isess, logits, image_input_tensors)
        mask_3d = gradient_saliency.GetMask(image_input)
    elif performance=='precise':
        gradient_saliency = saliency.IntegratedGradients(graph, isess, logits, image_input_tensors)
        mask_3d = gradient_saliency.GetSmoothedMask(image_input)
    else:
        raise NotImplementedError

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    mask_grayscale = saliency.VisualizeImageGrayscale(mask_3d)
    return mask_grayscale

def compute_metric(heatmap, mask, mode):
    hot_tresholds = [0.01*float(x) for x in range(100)]
    metrics = []
    if mode == 'SALIENCY':
        hot_masks = [heatmap > hot_treshold for hot_treshold in hot_tresholds]
        scaled_mask = cv2.resize(mask.astype(np.uint8), dsize=hot_masks[0].shape, interpolation=cv2.INTER_NEAREST)
        plt.imshow(scaled_mask)
        plt.show()
    elif mode == 'HEATMAP':
        [offset_x, offset_y] = [(x % stride) // 2 for x in img.shape[:2]]
        hot_masks = [heatmap < hot_treshold for hot_treshold in hot_tresholds]
    else:
        raise NotImplementedError

    for hot_mask in hot_masks:
        if mode == 'HEATMAP':
            hot_inside = 0
            inside_mask = np.zeros(hot_mask.shape).astype(np.bool)
            for i in range(hot_mask.shape[0]):
                for j in range(hot_mask.shape[1]):
                    x = i * stride + offset_x
                    y = j * stride + offset_y
                    x1 = max(0, x - grey_box_x // 2)
                    x2 = min(img.shape[0], x + grey_box_x // 2) 
                    y1 = max(0, y - grey_box_y // 2)
                    y2 = min(img.shape[1], y + grey_box_y // 2) 
                    if np.any(mask[x1 : x2, y1 : y2]) and hot_mask[i,j]:
                        hot_inside = hot_inside + 1
                    inside_mask[i,j] = np.any(mask[x1 : x2, y1 : y2])
        elif mode == 'SALIENCY':
            hot_inside = np.sum(np.logical_and(scaled_mask.astype(np.bool), hot_mask))
       
        #save_inside_mask = Image.fromarray((inside_mask*255).astype(np.uint8))
        #save_inside_mask.save(heatmap_path + basename + voc_name + '_inside_mask_box_'+ str(idx) +'.png')
        ratio = 0
        if hot_inside > 0:
            ratio = float(hot_inside) / np.sum(hot_mask)
        metrics.append(ratio)
    return metrics

def iou(boxA, boxB):
    # y1 x1 y2 x2
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
