import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
import tensorflow as tf
from imgProcessing import preProcessTrainImages
from random import uniform

import sys
sys.path.append(r"../../simulation")
from simulateMotionBlur import generateMotionBlurKernel
from simulateDefocusBlur import generateDefocusKernel

# These indices correspond approx. to spatial frequencies [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6] in lines/px.
FREQ_INDICES = (3, 6, 8, 11, 16, 22, 27, 33) 
IMG_PATCH_SIZE = 256
    
def rotateImg(img, imgPartPos, cameraResolution):
    """
    Rotate a PSF image according to is position in the camera sensor frame.
    See original work for more details.
    :param img: PSf tmage to rotate.
    :param imgPartPos: (Local) coordinates of the PSF image center.
    :param cameraResolution: Full resolution of the camera sensor frame.
    :return: Rotated PSF image according to its position inside the camera sensor frame.
    """
    enlargedImgPatchSize = len(img)

    # Get center coordinates of applied psf
    w = imgPartPos[0][0]
    h = imgPartPos[1][0]
    
    # Convert center to global image coord. system (origin at image center)
    w_ = - cameraResolution[0] / 2 + w
    h_ = cameraResolution[1] / 2 - h
    
    # Calculate rotation angle of psf.
    w_ = w_ if w_ != 0 else 1e-5
    angle = np.rad2deg(np.arctan(h_ / w_))
    if w_ < 0 and h_ > 0:
        angle = 180 + angle
    elif w_ < 0 and h_ < 0:
        angle = 180 + angle
    elif w_ > 0 and h_ < 0:
        angle = 360 + angle
    
    # Rotate psf image using calculated angle.
    patchLocalCenter = (enlargedImgPatchSize//2, enlargedImgPatchSize//2 )
    M = cv2.getRotationMatrix2D(patchLocalCenter, int(-angle), 1.0)
    rotated = cv2.warpAffine(img, M, (enlargedImgPatchSize, enlargedImgPatchSize))  
    return rotated
    
def generateGaussKernel(sigma, kernelSize=19):
    """
    Generates a 2D Gaussian kernel.
    :param sigma: Kernel standard deviation.
    :param kernelSize: Size of the generated kernel.
    :return: Generated 2D Gaussian Kernel.
    """

    x, y = np.mgrid[-kernelSize//2 + 1:kernelSize//2 + 1, -kernelSize//2 + 1:kernelSize//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def prepareTrainingData(imgs, pathOut, psfStats, imgInBitDepth, samplesPerImg, cameraResolution):
    """
    Prepare raw images to train the CNN for MTF estimation.
    :param imgs: Array of raw 1-channel images.
    :param pathOut: Path to the destination TFRecord file.
    :param psfStats: Real PSF images with corresponding metadata (to be published by the original work authors).
    :param imgInBitDepth: Bit depth of the input images (default: 16).
    :param samplesPerImg: Number of blur kernels to use per image (default: 6).
    :param cameraResolution: If psfStats is not None, you may specify the original camera resolution (default: [8688, 5792]).
    :return: Images (with PSFs applied) and corresponding label (MTF values) wrapped in a TFRecord file.
    """
    
    imgInMaxVal = 2 ** imgInBitDepth - 1
    imgOutMaxVal = 255.0
    writer = tf.io.TFRecordWriter(pathOut)
    # Iterate training images
    for imgIdx, img in enumerate(imgs):
        
        print("Image number:", imgIdx)
        if img.shape != (256, 256):
            continue
        
        # Normalize image to range [0,1]
        img = img / (2 ** imgInMaxVal - 1)
       
        # If there are PSFS given, pick some randomly. Otherwise, PSFs are simulated later.
        if psfStats is not None:
            psfStatsRandIdx = np.random.choice(range(len(psfStats)), samplesPerImg, replace=False)
            psfStatsRand = [psfStats[i, ...] for i in psfStatsRandIdx]
        else:
            psfStatsRand = range(samplesPerImg)
            
        for psfStat in psfStatsRand:
            
            # If there are PSFs, rotate them so that radial and tengential directions align with x and y directions.
            # Note: Not needed for estimatinos in horizontal and vertical image directions.
            if psfStats is not None:
                psfSensorPos = [psfStat[0], psfStat[1]]
                psf = psfStat[2]
                psf = rotateImg(psf, psfSensorPos, cameraResolution)
                psf = psf / np.sum(psf)
            
                # Combine PSF with random gaussian blur for more variety.
                if uniform(0, 1) <= 0.3:
                    randomSigma = uniform(0, 5)
                    randGaussKernel = generateGaussKernel(randomSigma)
                    psf = cv2.filter2D(psf, -1, randGaussKernel)
            # If there are no PSFs given, simulate random motion blur or defocus blur kernels.
            else:
                # According to PSF sizes of the original paper, to assure the correct MTF indices.
                kernelSize = 111 
                if uniform(0, 1) <= 0.5:
                    randomIntensity = uniform(0, 1)
                    psf = generateMotionBlurKernel(None, kernelSize, randomIntensity, asNumpyArray=True)
                else:
                    randomKernelDiameter = int(uniform(0, 31))
                    psf = generateDefocusKernel(randomKernelDiameter, kernelSize)
                    
            
            # Apply PSF to image and also get a 90° rotated version.
            blurredImg = cv2.filter2D(img, -1, psf)
            blurredImgRot = cv2.rotate(blurredImg, cv2.ROTATE_90_CLOCKWISE)
            
            # Extract MTF values from PSF in horizontal and vertical image directions.
            n = psf[0].size
            mtf = abs(fftshift(fft2(psf)))
            mtfRad = mtf[n//2, n//2:n-1]
            mtfTang= mtf[n//2:n-1, n//2]
            labelRad = [mtfRad[i] for i in FREQ_INDICES]
            labelTang = [mtfTang[i] for i in FREQ_INDICES]
            
            # Encode image as .png and append it with the horizontal MTF values to the TFRecord dataset.
            blurredImg = (blurredImg * imgOutMaxVal).astype("uint8")
            encoded_image_string = cv2.imencode(".png", blurredImg)[1].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_string])), 
                'y': tf.train.Feature(float_list=tf.train.FloatList(value=labelRad))
            }))
            writer.write(example.SerializeToString())
            
            # Encode the rotated image as .png and append it with the vertical MTF values to the TFRecord dataset.
            blurredImgRot = (blurredImgRot * imgOutMaxVal).astype("uint8")
            encoded_image_string = cv2.imencode(".png", blurredImgRot)[1].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_string])),
                'y': tf.train.Feature(float_list=tf.train.FloatList(value=labelTang))
            }))
            writer.write(example.SerializeToString())
            
    writer.close()
    print("Finished writing .tfrecord file.")
   

def prepareTFDataset(datasetEntry):
    """
    Unwraps image and corresponding label from a single TFRecord entry.
    :param datasetEntry: A single TFRecord entry.
    :return: Unwrapped image and corresponding label.
    """
    
    image_feature_description = {
        'X': tf.io.FixedLenFeature((), tf.string),
        'y': tf.io.FixedLenFeature((8,), tf.float32)
    }
    
    datasetEntry = tf.io.parse_single_example(datasetEntry, image_feature_description)
    img = datasetEntry['X']
    img = tf.image.decode_png(img, dtype=tf.uint8)
    img = tf.reshape(img, (IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    img = tf.py_function(preProcessTrainImages, [img, 3], tf.float32)
    
    label = datasetEntry['y']
    label = tf.expand_dims(label, axis=0)
    label = tf.expand_dims(label, axis=0)
    
    return img, label
    
def readTFRecordDataset(pathIn):
    """
    Unwraps dataset (images and corresponding labels) from a TFRecord file.
    :param pathIn: Path to the TFRecord dataset file.
    :return: Unwrapped dataset (images and corresponding labels).
    """
    
    dataset = tf.data.TFRecordDataset(pathIn)
    dataset = dataset.map(prepareTFDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
    
# Example to create training data  
# if __name__ == '__main__':       
#     imgs = np.load(r"/../../../photos.npz")["arr_0"] # Not part of this repository.
#     pathOut = r"../../../data/trainingDataset_defocusMotionBlur.tfrecord" # Not part of this repository.
#     prepareTrainingData(imgs, pathOut, None, 16.0, 6, [8688, 5792])
