import cv2
from scipy.fftpack import fft2, fftshift
import numpy as np
import os

# Indices that correspond to the following spatial MTF frequencies 
# (assuming an image size of 111x111 and 1.0 to be the Nyquist frequency):
# [0.08441692, 0.16883514, 0.22511481, 0.30953173, 0.45022833, 0.61906476, 0.75976135, 0.92859649])
FREQ_IDX = [3, 6, 8, 11, 16, 22, 27, 33]

def blurKernel2MTF(kernelImgPath, dirOut=None):
    """
    Extracts MTF values from a blur kernel estimation image.
    :param kernelImgPath: Path of the blur kernel estimation image.
    :param dirOut: Directory in which the results are stored.
    :return: None
    """
    
    # Create result dir
    if dirOut is not None and not os.path.exists(dirOut):
        os.makedirs(dirOut)
        
    # Read and normalize kernel image
    kernelImg = cv2.imread(kernelImgPath)[...,0]
    kernelImg = kernelImg / np.sum(kernelImg)
    w, h = kernelImg.shape
    
    # Extend kernel image to size 111x111 (to pick the same MTF frequencies as the CNN)
    size = 111
    centerPos = size // 2
    img = np.zeros((size, size))
    img[centerPos : centerPos + h, centerPos : centerPos + w] = kernelImg
    
    # Transform kernel image into Fourier space and get MTF
    mtf = abs(fftshift(fft2(img)))
    
    # Extract MTF values for desired frequencies in horizontal and vertical image directions
    resultsHor = np.zeros((1, 1, 8)) 
    resultsVert = np.zeros((1, 1, 8)) 
    for idx, freqIdx in enumerate(FREQ_IDX):
        resultsHor[0,0,idx] = mtf[centerPos, centerPos + freqIdx]
        resultsVert[0,0,idx] = mtf[centerPos + freqIdx, centerPos]
        
    # Save or print results
    if dirOut is not None:
        kernelImgName = kernelImgPath.split(os.sep)[-1].split(".")[0]
        pathResultsHor = os.path.join(dirOut, kernelImgName + "_MTF-H.npz")
        pathResultsVert = os.path.join(dirOut, kernelImgName + "_MTF-V.npz")
        if not os.path.exists(pathResultsHor) and not os.path.exists(pathResultsVert):
            np.savez_compressed(pathResultsHor, resultsHor)
            np.savez_compressed(pathResultsVert, resultsVert)
    else:
        print("MTF Hor:", resultsHor[0,0,:])
        print("MTF Vert:", resultsVert[0,0,:])
        
# # Example
# if __name__ == "__main__":
#     pathIn = r"..\data\udacity\labels_blur_patchwise\PMP\GT\1478020193690486111_car_1_0.jpg"     
#     dirOut = r"..\data\udacity\labels_blur_patchwise\PMP\GT\MTF"
    
#     if not os.path.exists(dirOut):
#         os.makedirs(dirOut)
        
#     blurKernel2MTF(pathIn, dirOut)