# Contact: Maik Wischow (maik.wischow@dlr.de), German Aerospace Center, Rutherfordstrasse 2, 12489 Berlin, Germany.

import numpy as np
import os
import math
import glob
import cv2

from buildModel import MTFNet
from imgProcessing import preProcessTestImages
from prepareTrainingData import readTFRecordDataset
import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

import sys
sys.path.append(r"../../../utils")
from objectDetectionUtils import getObjDetRoI, getPatchRoI, calculateIoU

IMG_PATCH_SIZE = 192
BASE_LR = 1e-4
TRAINING_BATCH_SIZE = 32
FRACTION_VALIDATION_DATA = 0.15
TRAINING_DATASET_BUFFER_SIZE = 1000
TRAINING_EPOCHS = 100
CNN_INPUT_SHAPE = (32,32,72)
IOU_THRESH = 0.7 # Only for function "estimateObjDetsMTF"

DEFAULT_MODEL_PATH = r"./model"
fileTypes = ['*.tiff', '*.png', "*.jpeg", "*.tif", "*.jpg", "*.gif"]

def prepareModel(modelPath = DEFAULT_MODEL_PATH):
    """
    Load or create a CNN model for MTF estimation.
    :param modelPath: Path to a CNN model to load.
    :return: Ready-to-use CNN model.
    """
    
    # Preparation
    tf.keras.backend.set_image_data_format('channels_last')
    keras.backend.clear_session()
    tf.python.framework.ops.reset_default_graph()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Load or create CNN model   
    if os.path.isfile(os.path.join(modelPath, "saved_model.pb")):
        M = tf.keras.models.load_model(modelPath)
        print("Successfully loaded checkpoint.")
    else:
        M = MTFNet(CNN_INPUT_SHAPE)
        print("No checkpoint found, created new model.")
        
    # Compile CNN model
    M.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
        loss="mean_squared_error",
        metrics=["mse"]
    )
    
    return M

def extractSubModelFC(mainModel, startIdx, endIdx):
    """
    Auxiliary function to extract the fully-connected sub-model.
    :param mainModel: The full CNN model.
    :param startIdx: Index of the first layer to extract.
    :param endIdx: Index of the last layer to extract.
    :return: Extracted fully-connected sub-model.
    """
    
    inputShape = mainModel.layers[startIdx].input.get_shape().as_list()[-3:]
    inputs = keras.Input(shape=inputShape)
    for idx in range(startIdx, endIdx + 1):
        if idx == startIdx:
           x = mainModel.layers[idx](inputs)
        else:
           x = mainModel.layers[idx](x)
      
    M = tf.keras.models.Model(inputs, x)
    return M

def defineAveragingModel(model):
    """
    Auxiliary function add an averaging layer to the CNN model.
    :param model: CNN model for MTF estimation.
    :return: CNN model including averaging layer.
    """
    
    # Extract sub-models for feature extraction and prediction
    subModelFeature = Model(inputs = model.layers[0].input, outputs = model.layers[-5].output)
    subModelPrediction = extractSubModelFC(model, 55, 58)
    
    # Add averaging layer and put all layers together
    inputShape = subModelFeature.layers[0].input.get_shape().as_list()[-3:]
    inputs = keras.Input(shape=inputShape)
    results = subModelFeature(inputs)
    avg = tf.keras.backend.mean(results, axis=0)
    output = subModelPrediction(avg)
    averagingModel = Model(inputs = inputs, outputs = output)
    
    return averagingModel

def saveMTFResult(MTF, dirOut, numHPatches, numWPatches, resultFileNamePrefix="", resultFileNameSuffix=".npz"):
    """
    Saves or prints a MTF estimation.
    :param MTF: MTF estimation to save.
    :param dirOut: Target directory to save the MTF esitmation.
    :param numHPatches: Number of image patches in vertical direction.
    :param numWPatches: Number of image patches in horizontal direction.
    :param resultFilenamePrefix: Name prefix of the file to save (default: "")
    :param resultFilenameSuffix: Name suffix of the file to save (default: ".npz")
    :return: None.
    """
    
    # Shape results
    resultsH = np.zeros((numHPatches, numWPatches, 8)) 
    resultsV = np.zeros((numHPatches, numWPatches, 8))
    for idx in range(len(MTF) // 2):
        hIdx = idx % numHPatches
        wIdx = idx // numHPatches
        resultsH[hIdx, wIdx] = MTF[2 * idx][0][0]
        resultsV[hIdx, wIdx] = MTF[2 * idx + 1][0][0]
            
    # Save results, if they not exist yet
    if dirOut is not None:
        if not os.path.exists(dirOut):
            os.makedirs(dirOut)
            
        pathMTFH = os.path.join(dirOut, resultFileNamePrefix + "_MTF-H" + resultFileNameSuffix)
        pathMTFV = os.path.join(dirOut, resultFileNamePrefix + "_MTF-V" + resultFileNameSuffix)
            
        if not os.path.exists(pathMTFH) and not os.path.exists(pathMTFV):
            np.savez_compressed(pathMTFH, resultsH)
            np.savez_compressed(pathMTFV, resultsV)
    else:
        print("MTF-H:", resultsH)
        print("MTF-V:", resultsH)

def estimateMTF(model, imgPathBatch, dirOut=None):
    """
    Apply a CNN model for MTF estimation.
    :param model: The (ready-to-use) CNN model.
    :param imgPathBatch: Array of images paths as input for the MTF estimation (default is a batch size of four images).
    :param dirOut: Directory to save the MTF estimation results.
    :return: None.
    """
    
    # Create the CNN model
    averagingModel = defineAveragingModel(model)
    
    # Load images and extract green channel for MTF estimation (according to the original paper)
    firstImageFileName = imgPathBatch[0].split(os.sep)[-1].split(".")[0]
    lastImageFileName = imgPathBatch[-1].split(os.sep)[-1].split(".")[0]
    imgBatch = [cv2.imread(imgPath) for imgPath in imgPathBatch]
    imgBatch = [img[..., 1] for img in imgBatch if img is not None and len(img.shape) == 3]
    imgBatch = np.array(imgBatch)
    if imgBatch is not None and len(imgBatch) > 0:
                
        # Pre-process images
        h, w = imgBatch[0].shape[0:2]
        numHPatches, numWPatches = math.ceil(h / IMG_PATCH_SIZE), math.ceil(w / IMG_PATCH_SIZE)
        imgs = np.array(preProcessTestImages(imgBatch, IMG_PATCH_SIZE))
        # Apply images to the CNN
        MTF = averagingModel(tf.convert_to_tensor(imgs), training=False).numpy()
        #Save and return results
        saveMTFResult(MTF, dirOut, numHPatches, numWPatches, str(firstImageFileName) + "-" + str(lastImageFileName))

def estimateObjDetsMTF(model, imgPathBatch, objDets, dirOut=None):   
    """
    Apply a CNN model for MTF estimation for object detection patches only.
    :param model: The (ready-to-use) CNN model.
    :param imgPathBatch: Array of images paths.
    :param dirOut: Directory to save the MTF estimation results.
    :param objDets: Array of ground truth object detections annotations. One annotation consists of five variables:
        objectClass, topLeftXCoordinate, topLeftYCoordinate, bottomRightXCoordinate, bottomRightYCoordinate
        (according to the keras-YOLOv3 format).
    :return: MTF Estimation results in horizontal and vertical image directions.
    """
    
    # Create CNN model
    averagingModel = defineAveragingModel(model)
    
    # Load images and extract green channel for MTF estimation (according to the original paper)
    imgBatch = [cv2.imread(imgPath) for imgPath in imgPathBatch]
    imgBatch = [img[..., 1] for img in imgBatch if img is not None]
    imgBatch = np.array(imgBatch)
    if imgBatch is not None and len(imgBatch) > 0:
        
        # Init cache
        imgShape = imgBatch[0].shape
        objDetRoICacheCoord = []
        objDetRoICacheResult = []
        # Iterate object detections
        for objDetIdx, objDet in enumerate(objDets):
            objClass, objx1, objy1, objx2, objy2 = objDet
            startX, startY, endX, endY = getObjDetRoI(imgShape, IMG_PATCH_SIZE, objx1, objy1, objx2, objy2)
            
            #Iterate corresponding image patches of detection
            numPatchesX = math.ceil((endX - startX) / IMG_PATCH_SIZE)
            numPatchesY = math.ceil((endY - startY) / IMG_PATCH_SIZE)
            for idxX in range(numPatchesX):
                for idxY in range (numPatchesY):
                    # Get image patch coordinates
                    startX_, startY_, endX_, endY_ = getPatchRoI(imgShape, IMG_PATCH_SIZE, startX, startY, idxX, idxY)
                    patchIdx = idxX * numPatchesY + idxY
                    resultFilenamePrefix = '_' + objClass + '_' + str(objDetIdx) + '_' + str(patchIdx)

                    # Search cache for fitting entry 
                    cacheHit = False;
                    for cacheIdx in range(len(objDetRoICacheCoord)):
                        x1, y1, x2, y2 = objDetRoICacheCoord[cacheIdx]
                        iou = calculateIoU(startX_, startY_, endX_, endY_, x1, y1, x2, y2)
                        if iou >= IOU_THRESH:
                            MTF = objDetRoICacheResult[cacheIdx];
                            saveMTFResult(MTF, dirOut, 1, 1, resultFilenamePrefix)
                            cacheHit = True;
                            break
                    
                    if cacheHit:
                        continue
                    
                    existingFiles = glob.glob(os.path.join(dirOut, "*" + resultFilenamePrefix + "*"))
                    if len(existingFiles) == 0:
                        imgPatchBatch = imgBatch[:, startY_:endY_, startX_:endX_]
                        h, w = imgShape[0:2]
                        
                        # Image pre-processing
                        imgs = preProcessTestImages(imgPatchBatch, IMG_PATCH_SIZE)
                        imgs = np.array(imgs)
                        
                        # Inference on image batch
                        MTF = averagingModel(tf.convert_to_tensor(imgs), training=False).numpy()
                        
                        # Save results
                        saveMTFResult(MTF, dirOut, 1, 1, resultFilenamePrefix)
                        objDetRoICacheCoord.append([startX_, startY_, endX_, endY_])
                        objDetRoICacheResult.append(MTF)
                            
    

def train(model, trainDataSetPath, checkpointSaveDir):
    """
    Train the CNN model for MTF esitmation.
    :param model: The untrained CNN model.
    :param trainDataSetPath: Path to the TFRecord file containt the training dataset.
    :param checkPointSaveDir: Target directory to store the checkpoint files during training.
    :return: None
    """
    
    # Load the TFRecord training dataset and split it in training and validation sub-sets.
    dataset = readTFRecordDataset(trainDataSetPath)
    datasetSize = 160500 # Number of samples in the training dataset. It is not a good style but it saves runtime, though.
    trainDataset = dataset.take(int(datasetSize * (1.0 - FRACTION_VALIDATION_DATA))).cache() \
                    .shuffle(buffer_size=TRAINING_DATASET_BUFFER_SIZE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    remaining = dataset.skip(int(datasetSize * (1.0 - FRACTION_VALIDATION_DATA)))
    validDataset = remaining.take(int(datasetSize * FRACTION_VALIDATION_DATA)).cache() \
                    .shuffle(buffer_size=TRAINING_DATASET_BUFFER_SIZE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Prepare model callbacks
    logdir = os.path.join(checkpointSaveDir, "logs")
    # Tensorboard callback
    tensordBoardCallback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Checkpoint callback.
    modelCheckpointCallback = ModelCheckpoint(
        filepath=checkpointSaveDir + "/weights-improvement-{epoch:02d}",
        save_weights_only=False,
        monitor='val_loss',
        save_best_only=False, 
        save_freq=1000
        )

    callbacks = [
        tensordBoardCallback,
        modelCheckpointCallback
    ]
    
    # Start the model training
    history = model.fit(trainDataset, epochs=TRAINING_EPOCHS, verbose=1, validation_data=validDataset, initial_epoch=0, shuffle=True, callbacks=callbacks)
    print(history.history)
    
    
# Example
# if __name__ == '__main__':
    
#     # Train the model
#     if False:
#         trainDataSetPath = r"../../../data/CNNTrainingData.TFRecord" # Not part of this repository.
#         checkPointSaveDir = r"./model/checkpoints"
#         M = prepareModel(checkPointSaveDir)
#         train(M, trainDataSetPath, checkPointSaveDir)
       
#     # Test the model
#     if True:
#         imgDirIn = r"../../../data/udacity/img/GT"
#         dirOut = r"../../../data/udacity/labels_blur_patchwise/CNN"
#         imgFileEnding = ".jpg"
#         imgPathBatch = glob.glob(os.path.join(imgDirIn, "*" + imgFileEnding))
#         M = prepareModel()
#         estimateMTF(M, imgPathBatch, None)