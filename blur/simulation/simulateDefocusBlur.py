import numpy as np
from scipy.signal import convolve
import cv2
import os

def generateDefocusKernel(diameter, kernelSize=33):
    """
    Generate a defocus kernel.
    :param diameter: Diameter of the actual generated kernel.
    :param kernelSize: Overall size of the kernel image in px.
    :return: Generated defocus blur kernel.
    """
    
    # Ensure uneven kernel diameter
    if diameter % 2 == 0:
        diameter += 1
    
    # Generate centered defocus blur kernel.
    kern = np.zeros((kernelSize, kernelSize), np.uint8)
    cv2.circle(kern, (kernelSize, kernelSize), diameter, 255, -1, cv2.LINE_AA, shift=1)
    
    # Normalize kernel to range [0,1]
    kern = np.float32(kern) / 255.0
    kern /= np.sum(kern)
    
    return kern
    
def addDefocusBlur(img, diameter, keep_image_dim=True):
    """
    Generate a defocus blur kernel and applied it an image.
    :param img: Input image to blur.
    :param diameter: Diameter of the generated defocus kernel.
    :param keep_image_dim: Keep the input image dimensions during the convolution of image and kernel.
    """

    # Decide convolution mode
    conv_mode = "valid"
    if keep_image_dim:
        conv_mode = "same"
    
    # Generate defocus blur kernel.
    kernel = generateDefocusKernel(int(diameter))
    resultChannels = ()
    numChannels = 3 if len(img.shape) == 3 else 1
    if numChannels > 1:
        for channelIdx in range(numChannels):
            # Convolve each image channel individually with the defocus kernel.
            resultChannel = convolve(img[:,:,channelIdx], kernel, mode=conv_mode).astype("uint8")
            # Collect blurred channels.
            resultChannels += resultChannel,
        result = np.dstack(resultChannels)
    else:
        result = convolve(img, kernel, mode=conv_mode).astype("uint8")
        
    return result


# Example
# if __name__ == '__main__':
#     dirIn = r"../../data/udacity/img/GT"
#     imgName = "1478019984182279255.jpg"
#     defocusDiameter = 11
#     dirOut = os.path.join(r"../../data/udacity/img/defocusBlur", str(defocusDiameter))
    
#     if not os.path.exists(dirOut):
#         os.makedirs(dirOut)
        
#     img = cv2.imread(os.path.join(dirIn, imgName))
#     defocusedImg = addDefocusBlur(img, 11)
#     cv2.imwrite(os.path.join(dirOut, imgName), defocusedImg)