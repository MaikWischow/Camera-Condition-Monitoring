import numpy as np
import os
import glob
import sys

sys.path.append(r"../../../utils")
from partImage import img2patches

IMG_PATCH_SIZE = 128

def rejectOutliers(data, m=3.5):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]
    
def evaluateNoiseEstimationMedianMinMax(dirsIn, pathLogFile, noiseLevels):
    """
    Evaluate robust (5% outliers rejected) median, min and max statistics of noise estimations.
    :param dirsIn: Main directories of noise estimation results. Each directory is supposed to contain 
                   sub-directories with noise estimation results (.npz) according to the applied noise levels (e.g., 0, 1, 2, ...).
    :param pathLogFile: Path to the output log file.
    :param noiseLevels: Applied ground truth noise levels when corrupting images.
    :return: Null
    """
    def writeLogEntry(logStream, entry):
        logStream.write(entry + "\n")
        print(entry)

    with open(pathLogFile, "w") as log:
        for dirIdx, dirPath in enumerate(dirsIn):
            writeLogEntry(log, "Method: " + dirPath)
            
            dirsInSplit = dirPath.rsplit(os.sep, 1)
            baseDir, corruptionTypeDir = dirsInSplit[-2], dirsInSplit[-1]
            medianByNoiseLevel = {key: [] for key in noiseLevels}
            minMaxErrorsByNoiseLevel = {key: [[], []] for key in noiseLevels}
            for noiseLevel in noiseLevels:
                noiseLevelStr = os.path.join(corruptionTypeDir, str(noiseLevel))
                resultFilPaths = glob.glob(os.path.join(baseDir, noiseLevelStr, "*.npz"))
                numResultFiles = 0
                
                writeLogEntry(log, "Noise Level: " + str(noiseLevel) + ", file count: " + str(len(resultFilPaths)) + ".")
                for idx, resultPath in enumerate(resultFilPaths):
                    try:
                        noiseResultMat = np.load(resultPath)["arr_0"]   
                    except: 
                        continue
                    patches = img2patches(noiseResultMat, IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
                    if len(patches) == 0 or len(patches.shape) != 3:
                        print("Skip result file", resultPath, "due to an invalid patch size.")
                        continue
                    
                    # Since the CNN estimates the noise level pixelwise, aggregate the estimations patch-wise using their median values first.
                    # Then, buffer min and max statistics.
                    patchMedians = np.median(patches, axis=(1,2))
                    minVal = np.min(patchMedians)
                    maxVal = np.max(patchMedians)
                    medianByNoiseLevel[noiseLevel] = np.concatenate((medianByNoiseLevel[noiseLevel], patchMedians), axis=None)
                    minMaxErrorsByNoiseLevel[noiseLevel][0].append(minVal)
                    minMaxErrorsByNoiseLevel[noiseLevel][1].append(maxVal)
                        
                    numResultFiles += 1
                    # For test purposes only: Take only the first x files into account.
                    # if idx == 10:
                    #     break
                        
                # Reject outliers of each statistic and calculate final statistics
                if numResultFiles != 0:
                    medianByNoiseLevel[noiseLevel]  = np.median(rejectOutliers(medianByNoiseLevel[noiseLevel]))
                    minMaxErrorsByNoiseLevel[noiseLevel][0] = np.min(rejectOutliers(minMaxErrorsByNoiseLevel[noiseLevel][0]))
                    minMaxErrorsByNoiseLevel[noiseLevel][1] = np.max(rejectOutliers(minMaxErrorsByNoiseLevel[noiseLevel][1]))
            
                # Write the results into the log file
                logLine = "Median: \n"
                if len(medianByNoiseLevel) > 0:
                    logLine += "(" + str(noiseLevel) + "," + str(medianByNoiseLevel[noiseLevel]) + ")"
                    writeLogEntry(log, logLine)
                    
                logLine = "Min: \n"
                if len(medianByNoiseLevel) > 0:
                    logLine += "(" + str(noiseLevel) + "," + str(minMaxErrorsByNoiseLevel[noiseLevel][0]) + ")"
                    writeLogEntry(log, logLine)
                    
                logLine = "Max: \n"
                if len(medianByNoiseLevel) > 0:
                    logLine += "(" + str(noiseLevel) + "," + str(minMaxErrorsByNoiseLevel[noiseLevel][1]) + ")"
                    writeLogEntry(log, logLine)
                writeLogEntry(log, "")
            writeLogEntry(log, "")
       
# Example
# noiseLevels = ["GT"] # [1,2,3,5,10,15,20,25,30]
# dirsIn = [r"..\..\..\data\udacity\labels_noise_patchwise\CNN",
#           r"..\..\..\data\udacity\labels_noise_patchwise\B+F",
#           r"..\..\..\data\udacity\labels_noise_patchwise\PCA"]
# pathLogFile = r"..\..\..\data\udacity\noiseLevelEstimationEvaluation.txt"
# evaluateNoiseEstimationMedianMinMax(dirsIn, pathLogFile, noiseLevels)