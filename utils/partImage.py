import numpy as np

def img2patches(img, imgPatchSizeX, imgPatchSizeY, stepSizeX, stepSizeY):
    """
    Slice image into patches.
    :param imgPatchSizeX: Patch size in x direction.
    :param imgPatchSizeY: Patch size in y direction.
    :param stepSizeX: Shift in x direction for next patch (no overlap between patches if stepSizeX==imgPatchSizeX).
    :param stepSizeY: Shift in y direction for next patch (no overlap between patches if stepSizeY==imgPatchSizeY).
    :return: 3-Channel image
    """
    results =  []
    for x in range(0, img.shape[0], stepSizeX):
        for y in range(0, img.shape[1], stepSizeY):
            temp = img[x : x + imgPatchSizeX, y : y + imgPatchSizeY, ...] 
            if temp.shape[0] == imgPatchSizeX and temp.shape[1] == imgPatchSizeY:
                results.append(temp)
    return np.array(results)