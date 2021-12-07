import cv2
import numpy as np
import math

ENLARGED_IMAGE_SIZE = 272
SUBSAMPLE_KERNEL_SIZE = 6
SOBEL_FILTER_KERNEL_SIZE = 3

def getLocalImagePatches(img, imgPatchSize, estimateBlurHorVert=True):
    """
    Part image in patches and rotate each image patch according to its image position (rotation applied only if estimateBlurHorVert==False).
    More details regarding the rotation can be found in the original paper: 
    Bauer, Matthias, et al. "Automatic estimation of modulation transfer functions." 2018 IEEE International Conference on Computational Photography (ICCP). IEEE, 2018.
    :param img: Image to part in patches.
    :param imgPatchSize: Target patch size (default: 192).
    :param estimateBlurHorVert: Flag if MTf is estimated in horizontal and vertical image directions.
    :return: Array of image patches.
    """
    
    results = [] 
        
    h, w = img.shape
    imgPatchPadding = (ENLARGED_IMAGE_SIZE - imgPatchSize) // 2.0
    maxW = math.ceil(w / imgPatchSize)
    maxH = math.ceil(h / imgPatchSize)
    for i in range(maxW):
        for j in range(maxH):
            
            # Calculate indices for image patch crop
            if i == 0:
                startI = 0
                endI = ENLARGED_IMAGE_SIZE
            elif i == maxW - 1:
                startI = w - ENLARGED_IMAGE_SIZE
                endI = w
            else:
                 startI = i * imgPatchSize - imgPatchPadding 
                 endI = i * imgPatchSize + imgPatchSize + imgPatchPadding 
                 
            if j == 0:
                startJ = 0
                endJ = ENLARGED_IMAGE_SIZE
            elif j == maxH - 1:
                startJ = h - ENLARGED_IMAGE_SIZE
                endJ = h
            else:
                 startJ = j * imgPatchSize - imgPatchPadding 
                 endJ = j * imgPatchSize + imgPatchSize + imgPatchPadding 
            
            patch = img[int(startJ) : int(endJ), int(startI) : int(endI)]
                
            # Calculate patch center point (global coord)
            x = startI + (ENLARGED_IMAGE_SIZE / 2)
            y = startJ + (ENLARGED_IMAGE_SIZE / 2)
            
            # Convert to different global coord. system (origin at image center)
            x_ = - w / 2 + x
            y_ = h / 2 - y
            
            # Get rotation angle (based on angle from polar coord.)
            if estimateBlurHorVert:
                angle = 0
            else:
                x_ = x_ if x_ != 0 else 1e-5
                angle = np.rad2deg(np.arctan(y_ / x_))
                if x_ < 0 and y_ > 0:
                    angle = 180 + angle
                elif x_ < 0 and y_ < 0:
                    angle = 180 + angle
                elif x_ > 0 and y_ < 0:
                    angle = 360 + angle
            
            # Rotate image using angle and crop center part (without black areas due to the rotation)
            patchLocalCenter = (ENLARGED_IMAGE_SIZE//2, ENLARGED_IMAGE_SIZE//2 )
            M = cv2.getRotationMatrix2D(patchLocalCenter, -angle, 1.0)
            rotated = cv2.warpAffine(patch, M, (ENLARGED_IMAGE_SIZE, ENLARGED_IMAGE_SIZE))  
            rotated = rotated[ \
                                patchLocalCenter[0] - imgPatchSize//2 : patchLocalCenter[0] + imgPatchSize//2, \
                                patchLocalCenter[1] - imgPatchSize//2 : patchLocalCenter[1] + imgPatchSize//2 \
                              ]
            
            results.append(rotated)
            
    return results
            

def subsampleImg(img):
    """
    Sub-sample images according to the original paper:
    Bauer, Matthias, et al. "Automatic estimation of modulation transfer functions." 2018 IEEE International Conference on Computational Photography (ICCP). IEEE, 2018.
    :param img: Image to be sub-sampled.
    """
    
    h, w = img.shape
    subsampled = [[] for i in range(SUBSAMPLE_KERNEL_SIZE * SUBSAMPLE_KERNEL_SIZE)]
    for i in range(h):
        for j in range(w):
            subsampled[((i % SUBSAMPLE_KERNEL_SIZE) * SUBSAMPLE_KERNEL_SIZE) + (j % SUBSAMPLE_KERNEL_SIZE)].append(img[i, j])
    
    results = []
    for s in subsampled:
        results.append(np.array(s).reshape((32, 32)))
        
    return results
           
def preProcessTestImages(imgs, imgPatchSize, estimateBlurHorVert=True):
    """
    Process raw (test) images so that they can be used as input for the CNN noise estimator.
    The images are prepared for estimations in horizontal and vertical image directions.
    :param imgs: Batch of images to be tested.
    :param imgPatchSize: Patch size of images according to the CNN (default: 192).
    :param estimateBlurHorVert: Flag if MTf is estimated in horizontal and vertical image directions (default: True).
    :return: Array of stacked cropped image patches and gradient images (each subsampled) to be used as input to test the CNN noise estimator.
    """
    
    resultImages = []
    for i, img in enumerate(imgs):
        try:
            if len(img.shape) == 3:
            # Get only green channel
                img = img[:, :, 1]
            img = (img - np.mean(img))
        except:
            continue
        
        # Divide image in patches (set angle = 0 to estimate blur in horizontal and vertical instead of 
        # radial and tangential directions)
        resultPatches = []
        h_, w_ = img.shape
        if h_ > ENLARGED_IMAGE_SIZE or w_ > ENLARGED_IMAGE_SIZE or not estimateBlurHorVert:
            # Only part images in patches if image is too large or a rotation needs to be applied
            localPatches = getLocalImagePatches(img, imgPatchSize, ENLARGED_IMAGE_SIZE)
        else:
            localPatches = [img]
        for localPatch in localPatches:
        
            # Rotate image
            localPatchRot = cv2.rotate(localPatch, cv2.ROTATE_90_CLOCKWISE)
            
            # Calculate edge images
            localPatchGrad = cv2.Sobel(localPatch, cv2.CV_64F, 1, 0, ksize=SOBEL_FILTER_KERNEL_SIZE, borderType=cv2.BORDER_REFLECT)
            localPatchRotGrad = cv2.Sobel(localPatchRot, cv2.CV_64F, 1, 0, ksize=SOBEL_FILTER_KERNEL_SIZE, borderType=cv2.BORDER_REFLECT)
            
            # Subsampling (6x6 filter)
            localPatchSubsampled = subsampleImg(localPatch)
            localPatchRotSubsampled = subsampleImg(localPatchRot)
            localPatchGradSubsampled = subsampleImg(localPatchGrad)
            localPatchRotGradSubsampled = subsampleImg(localPatchRotGrad)
            
            # Combine list to desired shape (32x32x72)
            localPatchSubsampled.extend(localPatchGradSubsampled)
            localPatchRotSubsampled.extend(localPatchRotGradSubsampled)
            
            # Change dimension from "channels first" to "channels last" and add to results
            localPatchSubsampled = np.moveaxis(np.array(localPatchSubsampled), 0, 2)
            localPatchRotSubsampled = np.moveaxis(np.array(localPatchRotSubsampled), 0, 2)
            
            # Normalize range to [0,1] (as in training)
            localPatchSubsampled = localPatchSubsampled.astype(np.float32) / 255.0
            localPatchRotSubsampled = localPatchRotSubsampled.astype(np.float32) / 255.0
            resultPatches.append(localPatchSubsampled)
            resultPatches.append(localPatchRotSubsampled)
            
        resultImages.append(resultPatches)
            
    return np.array(resultImages)

def preProcessTrainImages(img):
    """
    Process raw (training) image patches so that they can be used as input for the CNN noise estimator.
    :param img: Raw input image patch.
    :return: Array of stacked cropped image patch and gradient image (each subsampled) to be used as input to train the CNN noise estimator.
    """
          
    # Crop image to size 192x192 and mean normalize it
    img = img[0:192, 0:192]
    img = img.numpy().astype(np.uint8)
    img = (img - np.mean(img))
    
    # Calculate grad image
    imgGrad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=int(SOBEL_FILTER_KERNEL_SIZE), borderType=cv2.BORDER_REFLECT)
    # Subsampling image and grad image with a 6x6 filter
    imgSubsampled = subsampleImg(img)
    imgGradSubsampled = subsampleImg(imgGrad)
    
    # Combine list to desired shape (32x32x72)
    imgSubsampled.extend(imgGradSubsampled)
    # Change dimension from "channels first" to "channels last"
    imgSubsampled = np.moveaxis(np.array(imgSubsampled), 0, 2)
    # Normalize intensity range to [0,1]
    imgSubsampled = imgSubsampled.astype(np.float32) / 255.0
    return imgSubsampled          