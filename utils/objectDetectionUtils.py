def getObjDetRoI(imgSize, imgPatchSize, objx1, objy1, objx2, objy2):
    """
    Get region of interest (ROI) for a given object detection with respect to image and image patch boundaries.
    :param imgSize: size of the image of interest (e.g., [1920x1080]).
    :param imgPatchSize: Patch size of the image patch of interest (e.g., 192).
    :param objx1: Upper left x coordinate of the object detection.
    :param objy1: Upper left y coordinate of the object detection.
    :param objx2: Lower right x coordinate of the object detection.
    :param objy2: Lower right y coordinate of the object detection.
    """
    
    # Cast to float values for calculations
    startX = float(objx1);
    startY = float(objy1);
    endX = float(objx2);
    endY = float(objy2);
    
    # Ensure image and image patch boundaries
    xRange = endX - startX;
    yRange = endY - startY;
    addX = (imgPatchSize - (xRange % imgPatchSize));
    addY = (imgPatchSize - (yRange % imgPatchSize));
    endX = endX + addX;
    endY = endY + addY;
    
    if endX > imgSize[1]:
        endX = imgSize[1]
    if endY > imgSize[0]:
        endY = imgSize[0]
        
    return startX, startY, endX, endY

def getPatchRoI(imgShape, imgPatchSize, startX, startY, idxX, idxY):
    """
    Get region of interest (ROI) for a given patch index with respect to image patch boundaries.
    :param imgShape: size of the image of interest (e.g., [1920x1080]).
    :param imgPatchSize: Patch size of the image patch of interest (e.g., 192).
    :param startX: Start x coordinate.
    :param startY: Start y coordinate.
    :param idxX: Patch index in x direction.
    :param idxY: Patch index in y direction.
    """
    
    # Ensure image patch boundaries
    startX_ = startX + idxX * imgPatchSize;
    startY_ = startY + idxY * imgPatchSize;
    endX_ = startX_ + imgPatchSize;
    endY_ = startY_ + imgPatchSize;
    if endX_ > imgShape[1]:
        offset = endX_ - imgShape[1];
        endX_ = endX_ - offset; 
        startX_ = startX_ - offset;
    if endY_ > imgShape[0]:
        offset = endY_ - imgShape[0];
        endY_ = endY_ - offset; 
        startY_ = startY_ - offset;
        
    return int(startX_), int(startY_), int(endX_), int(endY_)


def calculateIoU(startX_, startY_, endX_, endY_, x1, y1, x2, y2):
    """
    Calculate the intersection over union (IoU) score for two bounding boxes.
    :param startX_: Upper left x coordinate of the first object detection.
    :param startY_: Upper left y coordinate of the first object detection.
    :param endX_: Lower right x coordinate of the first object detection.
    :param endY_: Lower right y coordinate of the first object detection.
    :param x1: Upper left x coordinate of the second object detection.
    :param y1: Upper left y coordinate of the second object detection.
    :param x2: Lower right x coordinate of the second object detection.
    :param y2: Lower right y coordinate of the second object detection.
    """
    
    x_left = max(startX_, x1);
    y_top = max(startY_, y1);
    x_right = min(endX_, x2);
    y_bottom = min(endY_, y2);

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top);
    bb1_area = (endX_ - startX_) * (endY_ - startY_);
    bb2_area = (x2 - x1) * (y2 - y1);
    iou = intersection_area / (bb1_area + bb2_area - intersection_area);
    return iou