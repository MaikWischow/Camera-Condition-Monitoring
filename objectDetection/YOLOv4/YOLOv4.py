import cv2
import numpy as np
import os
import onnxruntime as rt
from scipy import special
import glob

PATH_MODEL = r"./model/yolov4.onnx" # Not included in GitHub repository
PATH_CLASSES_FILE = r"./model/coco.names"
PATH_YOLO_ANCHORS = r"./model/yolov4_anchors.txt"
IMG_FILE_TYPES = ['*.tiff', '*.png', '*.jpg', "*.jpeg", '*.gif'] 
IMG_INPUT_SIZE = 416
IOU_THRESHOLD = 0.5

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes

def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    '''define anchor boxes'''
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def detectObjects(dirInImgs, dirOutResults, subDirs=[""], numThreads=8):
    """
    Apply YOLOv4 object detection on images found in dirInImgs and save results to .txt files.
    :param dirInImgs: Input directory.
    :param dirOutResults: Output directory.
    :param subDirs: Sub-directories of "dirInImgs" to search for images (default: [""]).
    :param numThreads: OnnxOptions parameter, number of CPU threads for multi-threadding.
    """
    
    # Prepare session options
    onnxOptions = rt.SessionOptions()
    onnxOptions.intra_op_num_threads = numThreads
    onnxOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Prepare YOLOv4 model
    sess = rt.InferenceSession(PATH_MODEL, onnxOptions)
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name
    
    # Iterate subdirs of dirInImgs
    for subDir in subDirs:
        
        subDir = str(subDir)
        dirIn = os.path.join(dirInImgs, subDir)
        dirOut = os.path.join(dirOutResults, subDir)
        
        if not os.path.exists(dirIn):
            continue
        if not os.path.exists(dirOut):
            os.makedirs(dirOut)
    
        # iterate image file extensions
        for fileType in IMG_FILE_TYPES:
            imgPaths = glob.glob(os.path.join(dirIn, fileType))
            for imgPath in imgPaths:
                
                fileName = imgPath.split(os.sep)[-1].split(".")[0]
                resultFileName = os.path.join(dirOut, fileName) + ".txt"
                if not os.path.exists(imgPath) or os.path.exists(resultFileName):
                    continue
                
                # Load image and convert to RGB
                try:
                    original_image = cv2.imread(imgPath)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    original_image_size = original_image.shape[:2]
                except:
                    continue
        
                # Pre-process images (from original source code)
                image_data = image_preprocess(np.copy(original_image), [IMG_INPUT_SIZE, IMG_INPUT_SIZE])
                image_data = image_data[np.newaxis, ...].astype(np.float32)
        
                # Apply YOLOv4 model on the images
                detections = sess.run(output_names, {input_name: image_data})
        
                # Post-process object detections (from original source code)
                STRIDES = [8, 16, 32]
                XYSCALE = [1.2, 1.1, 1.05]
                ANCHORS = get_anchors(PATH_YOLO_ANCHORS)
                STRIDES = np.array(STRIDES)
                pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
                bboxes = postprocess_boxes(pred_bbox, original_image_size, IMG_INPUT_SIZE, IOU_THRESHOLD)
                bboxes = nms(bboxes, 0.213, method='nms')
        
                # Write object detections to file
                with open(os.path.join(dirOut, fileName) + ".txt", 'w') as resultsFile:
                    for i, bbox in enumerate(bboxes):
                        classes = read_class_names(PATH_CLASSES_FILE) 
                        className = classes[int(bbox[5])]
                        resultsFile.write(
                            str(className) \
                            + " " + str(float(bbox[4])) \
                            + " " + str(int(bbox[0])) \
                            + " " + str(int(bbox[1])) \
                            + " " + str(int(bbox[2])) \
                            + " " + str(int(bbox[3])) \
                            + '\n')
                    resultsFile.close()
        
    return

# Example
# if __name__ == '__main__':
#     dirIn = r"../../data/udacity/img"
#     dirOut = r"../../data/udacity/labels_object_detection/YOLOv4"
#     detectObjects(dirIn, dirOut, subDirs=["GT"])