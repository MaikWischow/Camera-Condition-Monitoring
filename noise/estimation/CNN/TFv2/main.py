import sys
import os
import tqdm
import cv2
import numpy as np
from random import uniform

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

sys.path.append("../../../Simulation")
from generateDateset import DatabaseCreator
from generateHighLevelNoise import applyPhotonDarkReadNoise

# Define image parameters
IMG_PATCH_SHAPE = 128
MAX_IMG_INTENSITY = 255.0

# Define training parameters
NOISE_SAMPLES_PER_IMG = 3
BASE_LR = 1e-4
BATCH_SIZE = 1
VAL_DATA_SPLIT = 0.10

# Define the neural network model parameters
FEATURE_DIM = 64
MODEL_DEPTH = 12
RESIDUAL_BLOCK_DEPTH = 4
NUM_RESIDUAL_BLOCKS = MODEL_DEPTH // RESIDUAL_BLOCK_DEPTH
CONV_KERNEL_SIZE = [3, 3]

def build_model():
    """
    Setup the raw CNN model.
    :return: Raw CNN model.
    """
    
    def residual_block(h, width, kernelSize, depth):
        h_in = h
        for i in range(depth):
            h = tf.keras.layers.Conv2D(width, kernelSize, padding='same', activation='relu')(h)
        return h_in + h
    
    # Setup the NN model
    x = tf.keras.Input(name="x", shape=(IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 3), dtype=tf.dtypes.float32)
    h = x
    for idx in range(0, NUM_RESIDUAL_BLOCKS):
        h = tf.keras.layers.Conv2D(FEATURE_DIM, CONV_KERNEL_SIZE, padding='same',  activation='relu')(h)
        h = residual_block(h, FEATURE_DIM, CONV_KERNEL_SIZE, RESIDUAL_BLOCK_DEPTH)
    h = tf.keras.layers.Conv2D(1 , CONV_KERNEL_SIZE, padding='same', activation='relu')(h)
    
    return tf.keras.Model(x, h, name="tfv2_keras_model")

def load_model(modelDir):
    """
    Setup a raw CNN model and load trained weights from the modelDir.
    :param modelDir: Path to the trained weights.
    :return: If weights could be found in modelDir: CNN with trained weights. Otherwise: Untrained CNN.
    """
    
    def custom_loss(y_true, y_pred):
        lmd = 0.25
        
        cost_mat = tf.reduce_sum(tf.square(tf.subtract(y_true,  y_pred))) / BATCH_SIZE
        cost_scalar = tf.square(tf.subtract(tf.reduce_mean(y_pred), tf.reduce_mean(y_true)))

        cost = lmd * cost_mat + (1.0 - lmd) * cost_scalar
        return cost  

    model = build_model()
    try:
        model.load_weights(modelDir)
        print("Successfully load model weights.")
    except:
        print("Warning: No model weights found. Continue with untrained model.")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
        loss=custom_loss,
        metrics=["mse"]
    )
    
    return model
    
def train(model, modelPath, trainPath, valPath, minNoiseLevel, maxNoiseLevel):
        """
        Training using Estimator class.
        :param model: Compiled CNN model.
        :param modelPath: path to save trained model+weights
        :param trainPath: path to training dataset
        :param valPath: path to validation dataset
        :param minNoiseLevel: minimum noise level in [0,255] DN added to clean images
        :param maxNoiseLevel: maximum noise level in [0,255] and greater than minNoiseLevel added to clean images
        :return: -
        """
                
        # Load the test and validation dataset (preserved compatibility to datasets of previous version)
        dc = DatabaseCreator()
        valData = dc.load_hdf5_v1(valPath, 'rgb')
        valData = np.array(valData)
        trainData = dc.load_hdf5_v1(trainPath, 'rgb')
        trainData = np.array(trainData).astype(np.float32)
        combinedTrainingData = np.concatenate((trainData, valData))[0:5]
        
        # Generate and apply noise to train and val data, "NOISE_SAMPLES_PER_IMG" times per image.
        data = []
        labels = []
        for imgPatch in combinedTrainingData:
            for i in range(NOISE_SAMPLES_PER_IMG):
                grayImgPatch = cv2.cvtColor(imgPatch.astype("uint8"), cv2.COLOR_RGB2GRAY)
                grayImgPatch = cv2.merge((grayImgPatch, grayImgPatch, grayImgPatch))
                grayImgPatch = grayImgPatch.astype("float32") / MAX_IMG_INTENSITY 
                
                randNoiseLevel = uniform(minNoiseLevel, maxNoiseLevel)
                noiseMap = np.zeros((IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 1), dtype=np.float32)
                noiseMap.fill(randNoiseLevel / MAX_IMG_INTENSITY)
                labels.append(noiseMap)
                
                grayNoised, _ = applyPhotonDarkReadNoise(grayImgPatch, randNoiseLevel)
                grayNoised = grayNoised.astype("float32") / MAX_IMG_INTENSITY
                grayNoised = cv2.merge((grayNoised, grayNoised, grayNoised))
                data.append(grayNoised)
            
        data = np.array(data) 
        labels = np.array(labels)
        combinedTrainingData = []
         
        # Save model+weights every 2 epochs
        modelCheckpointCallback = ModelCheckpoint(
            filepath=modelPath + "/weights-improvement-{epoch:02d}",
            save_weights_only=False,
            monitor='val_loss',
            save_best_only=False, 
            save_freq='epoch',
            period=2
        )
        
        # Train the model. You may also use separated train and val data sets.
        model.fit(data, labels, epochs=100, verbose=1, validation_split=VAL_DATA_SPLIT, initial_epoch=0, shuffle=True, callbacks=[modelCheckpointCallback], batch_size=BATCH_SIZE)
        model.save_weights(modelPath)

def estimateNoise(model, imgPath, dirOut=None):
    """
    Estimate noise of an image patch-wise.
    :param model: Loaded CNN model with trained weights.
    :param imgPath: Path to the image to assess.
    :param dirOut: dirOut!=None: Noise estimation results will be saved in this directory. Otherwise: No results will be saved.
    :return noiseEstimation: Noise estimation image of size [IMG_PATCH_SHAPE, IMG_PATCH_SHAPE].
    """
    
    # Load and convert image to range [0,1]
    img = np.array(cv2.imread(imgPath))
    img = img / MAX_IMG_INTENSITY
        
    assert img.ndim == 3
    h, w = img.shape[:2]
    
    # Determine image patch size
    psize = IMG_PATCH_SHAPE
    psize = min(min(psize, h), w)
    psize -= psize % 2
    patch_step = psize
    shift_factor = 2

    # Result array
    noiseEstimation = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float32)

    rangex = range(0, w, patch_step)
    rangey = range(0, h, patch_step)
    ntiles = len(rangex) * len(rangey)
    with tqdm.tqdm(total=ntiles, unit='tiles', unit_scale=True) as pbar:
        for start_x in rangex:
            for start_y in rangey:

                # Calculateimg next patch coordinates
                end_x = start_x + psize
                end_y = start_y + psize
                if end_x > w:
                    end_x = w
                    end_x = shift_factor * ((end_x) // shift_factor)
                    start_x = end_x - psize
                if end_y > h:
                    end_y = h
                    end_y = shift_factor * ((end_y) // shift_factor)
                    start_y = end_y - psize

                # Estimate image noise
                tileM = img[np.newaxis, start_y:end_y, start_x:end_x, :]
                result = model(tf.convert_to_tensor(tileM), training=False).numpy()
                result = result.reshape(psize, psize, 1)
                noiseEstimation[start_y:start_y + psize, start_x:start_x + psize, :] = result
                pbar.update(1)
    
    # Ensure img intensity ranges and convert to range [0, MAX_IMG_INTENSITY]
    noiseEstimation[noiseEstimation < 0.0] = 0.0
    noiseEstimation[noiseEstimation > 1.0] = 1.0
    noiseEstimation *= MAX_IMG_INTENSITY
    noiseEstimation = noiseEstimation.astype("uint8")
    
    # Save the result as .npz file (but do not override existing files)
    if dirOut is not None:
        if not os.path.exists(dirOut):
            os.makedirs(dirOut)
        imgName = imgPath.split(os.sep)[-1].split(".")[0]
        noiseMapPath = os.path.join(dirOut, imgName + ".npz")
        if not os.path.exists(noiseMapPath):
            np.savez_compressed(noiseMapPath, noiseEstimation[..., 0])
        else:
            print("The noise estimation result file", noiseMapPath, "already exists. Did not override the result.")

    return noiseEstimation

# # Example
# if __name__ == '__main__':
    
#     # Set GPU and allow dynamic memory growth
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     # Setup clear tf and keras sessions
#     tf.keras.backend.clear_session()
#     tf.python.framework.ops.reset_default_graph()
#     tf.keras.backend.set_image_data_format('channels_last')

    ## Train the model
    # if False:
    #     trainDatasetPath = r".../train.h5" # Not included in this repository.
    #     testDatasetPath = r"...test.h5" # Not included in this repository.
       
    #     modelDir = r".\newModel"
    #     if not os.path.exists(modelDir):
    #         os.makedirs(modelDir)

    #     minNoiseLevel = 0.01
    #     maxNoiseLevel = 30.0
    #     model = load_model(modelDir)
    #     train(model, modelDir, trainDatasetPath, testDatasetPath, minNoiseLevel, maxNoiseLevel)
       
    ## Test the model
    # if False:
    #     import glob
    #     dirIn = r"../../../../data/udacity/img/noised/25"
    #     imgFileEnding = ".jpg"
    #     modelDir = r".\model\variables\variables"
    #     model = load_model(modelDir)
    #     for imgPath in glob.glob(os.path.join(dirIn, "*" + imgFileEnding)):
    #         noiseMap = estimateNoise(model, imgPath, None)
    #         print("Median noise level:", np.median(noiseMap))
    #         print("Mean noise level:", np.mean(noiseMap))