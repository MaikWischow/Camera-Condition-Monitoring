import numpy as np
from PIL import Image
import onnxruntime as rt
import os
import glob

PATH_MODEL = r"./model/FasterRCNN-10.onnx" # Not included in GitHub repository
PATH_CLASSES_FILE = r"./model/coco_classes.txt"
IMG_FILE_TYPES = ['*.tiff', '*.png', '*.jpg',"*.jpeg", "*.gif"] 
IMG_INPUT_SIZE = 800.0
IOU_THRESHOLD = 0.5

def preProcess(image):
    # Resize
    ratio = IMG_INPUT_SIZE / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to RGB (if grayscale)
    image = image.convert('RGB')
    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image 
 
def detectObjects(dirInImgs, dirOutResults, subDirs=[""], numThreads=8):
    """
    Apply Faster-RCNN object detection on images found in dirInImgs and save results to .txt files.
    :param dirInImgs: Input directory.
    :param dirOutResults: Output directory.
    :param subDirs: Sub-directories of "dirInImgs" to search for images (default: [""]).
    :param numThreads: OnnxOptions parameter, number of CPU threads for multi-threadding.
    """
    
    # Prepare session options
    onnxOptions =rt.SessionOptions()
    onnxOptions.intra_op_num_threads = 15
    onnxOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Prepare YOLOv4 model
    sess = rt.InferenceSession(PATH_MODEL, onnxOptions)
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name
    classes = [line.rstrip('\n') for line in open(PATH_CLASSES_FILE)]
    
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
                    # Load nad pre-process images (from original source code)
                    img = Image.open(imgPath)
                    preProcessedImg = preProcess(img)
                except:
                    continue
                
                # Apply Faster-RCNN on image
                detections = sess.run(output_names, {input_name: preProcessedImg})
                boxes, labels, scores = detections
                
                # Resize object detection bounding boxes
                ratio = IMG_INPUT_SIZE / min(img.size[0], img.size[1])
                boxes /= ratio
                
                # Write object detections to file
                with open(os.path.join(dirOut, fileName) + ".txt", 'w') as resultsFile:
                    for box, label, score in zip(boxes, labels, scores):
                        if score >= IOU_THRESHOLD:
                            className = classes[label]
                            resultsFile.write( \
                                      str(className) \
                                      + " " + str(float(score)) \
                                      + " " + str(int(box[0])) \
                                      + " " + str(int(box[1])) \
                                      + " " + str(int(box[2])) \
                                      + " " + str(int(box[3])) \
                                      + '\n')
                    resultsFile.close()
                    
    return

# Example
# if __name__ == '__main__':
#     dirIn = r"../../data/udacity/img"
#     dirOut = r"../../data/udacity/labels_object_detection/faster-RCNN"
#     detectObjects(dirIn, dirOut, subDirs=["GT"])