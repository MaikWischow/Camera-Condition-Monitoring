import numpy as np
import os
import glob
from sklearn.metrics import mean_absolute_error

# Approx. spatial MTF frequencies in lines/px we use for evaluation.
MTF_FREQUENCIES_PX = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])

def reject_outliers(data, m=3.5):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]
    
def evaluateMTFEstimations(dirInMtfResult, mtfDirection, groundTruthMTF, pathLogFile):
    """
    Evaluate MTF estimations using the mean absolute error metric and given ground truth values.
    :param dirInMtfResult: Directory with the MTF estimation results.
    :param mtfDirection: Which image direction shall be evaluated (H, V, or H+V)?
    :param groundTruthMTF: Dictionary with the ground truth MTF values at the stated spatial frequencies (MTF_FREQUENCIES_PX) of the applied blur kernel.
        Example for linear motion blur with size 11px: {"H" : [0.991,0.964,0.937,0.885,0.779,0.637,0.52,0.387],
                                                        "V" : [0.871,0.542,0.283,0.046,0.207,0.047,0.122,0.043] }
    """
    
    def writeLogEntry(logStream, entry):
        logStream.write(entry + "\n")
        print(entry)
        
    if groundTruthMTF is None or dirInMtfResult is None or groundTruthMTF is None:
        raise Exception("There is min. one input parameter which is None.")
    if (mtfDirection == "V" or mtfDirection == "H") and \
        (mtfDirection not in groundTruthMTF.keys() or len(groundTruthMTF[mtfDirection]) == 0):
        raise Exception("Ground truth values expected for the MTF direction:", mtfDirection)
    
    # Find the MTF estimation result file paths for the desired image direction (H+V is the average of both directions).
    if "H+V" == mtfDirection:
         mtfResultPathsH = glob.glob(os.path.join(dirInMtfResult, "*" +"MTF" + "*" + "H" + "*.npz"))
         mtfResultPathsV = glob.glob(os.path.join(dirInMtfResult, "*" +"MTF" + "*" +"V" + "*.npz"))
    elif "H" == mtfDirection:
         mtfResultPathsH = glob.glob(os.path.join(dirInMtfResult, "*" +"MTF" + "*" + "H" + "*.npz"))
         mtfResultPathsV = np.zeros(len(mtfResultPathsH))
    elif "V" == mtfDirection:
         mtfResultPathsV = glob.glob(os.path.join(dirInMtfResult, "*" +"MTF" + "*" + "V" + "*.npz"))
         mtfResultPathsH = np.zeros(len(mtfResultPathsV))
    else:
         raise Exception("Unsupported MTF direction:", mtfDirection)
        
    # Load the MTF results.
    # Since we only get MTF estimations of object detection patches (size 192x192) into account, the result is always at matrix position [x,y] ==[0,0].
    numFiles = 0
    resultMTFs = [[], []]
    for (mtfResultPathH, mtfResultPathV) in zip(mtfResultPathsH, mtfResultPathsV):
        for idx, mtfResult in enumerate([mtfResultPathH, mtfResultPathV]):
            if mtfResult != 0:
                resultMTFMat = np.load(mtfResult)["arr_0"] 
                if len(resultMTFMat) > 0:
                    resultMTFs[idx].append(resultMTFMat[0, 0]) 
            
        numFiles += 1
        # Stop after a given number of result files (for test purposes only)
        # if numFiles == 1000:
        #     break
    
    # Calculate the desired (robust) statistics (median, min and max).
    robustMTF = [[], []]
    medianMTF = [[], []]
    minMTF = [[], []]
    maxMTF = [[], []]
    for resultMtfIdx in range(len(resultMTFs)):
        resultMTFsByDirection = np.array(resultMTFs[resultMtfIdx])
        if len(resultMTFsByDirection) > 0:
            if len(resultMTFsByDirection) > 1:
                robustMTF = [reject_outliers(resultMTFsByDirection[:,i]) for i in range(8)]
                medianMTF[resultMtfIdx] = np.array([np.median(robustMTF[i]) for i in range(8)])
                minMTF[resultMtfIdx] = np.array([np.min(robustMTF[i]) for i in range(8)])
                maxMTF[resultMtfIdx] = np.array([np.max(robustMTF[i]) for i in range(8)])
            elif len(resultMTFsByDirection) == 1:
                medianMTF[resultMtfIdx] = minMTF[resultMtfIdx] = maxMTF[resultMtfIdx] = resultMTFsByDirection[0]
            else:
                raise Exception("No MTF results found.")
    
    with open(pathLogFile, "w") as log:
        # Print results
        writeLogEntry(log, "DirIn:" +  dirInMtfResult)
        mtfH = medianMTF[0]
        minH = minMTF[0]
        maxH = maxMTF[0]
        mtfV = medianMTF[1]
        minV = minMTF[1]
        maxV = maxMTF[1]
        
        # Averaged results from horizontal and vertical directions.
        if "H+V" == mtfDirection:
            writeLogEntry(log, "H+V")
            if len(mtfH) > 0 and len(mtfV) > 0:
               maeH = mean_absolute_error(groundTruthMTF["H"], mtfH)
               maeV = mean_absolute_error(groundTruthMTF["V"], mtfV)
               mae = (maeH + maeV) / 2.0
               writeLogEntry(log, "MEA of median (H+V): " + str(mae))
        
        # Results in horizontal direction.
        elif "H" == mtfDirection:
            writeLogEntry(log, "H:")
            if len(mtfH) > 0:
                writeLogEntry(log, "Median:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, mtfH):
                  logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["H"], mtfH)
                writeLogEntry(log, "MEA of median: " + str(mae))
                    
            if len(minH) > 0:
                writeLogEntry(log, "Min:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, minH):
                   logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["H"], minH)
                writeLogEntry(log, "MEA of min: " + str(mae))
                
            if len(maxH) > 0:
                writeLogEntry(log, "Max:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, maxH):
                   logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["H"], maxH)
                writeLogEntry(log, "MEA of max: " + str(mae))
        
        # Results in vertical direction.
        elif "V" == mtfDirection:
            writeLogEntry(log, "V:")
            if len(mtfV) > 0:
                writeLogEntry(log, "Median:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, mtfV):
                  logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["V"], mtfV)
                writeLogEntry(log, "MEA of median: " + str(mae))
                
            if len(minV) > 0:
                writeLogEntry(log, "Min:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, minV):
                   logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["V"], minV)
                writeLogEntry(log, "MEA of min: " + str(mae))
                
            if len(maxV) > 0:
                writeLogEntry(log, "Max:")
                logLine = ""
                for (freq, mtf) in zip(MTF_FREQUENCIES_PX, maxV):
                   logLine += "(" + str(freq) + "," + str(round(mtf, 3)) + ")"
                writeLogEntry(log, logLine)
                mae = mean_absolute_error(groundTruthMTF["V"], maxV)
                writeLogEntry(log, "MEA of max: " + str(mae))
         

# Example
# if __name__ == '__main__':  
#     dirInMtfResult = r"../../../data/udacity/labels_blur_patchwise/PMP/GT/MTF"
#     pathLogFile = r"../../../data/udacity/MTFEstimationEvaluation.txt"
#     mtfDirection="H"
#     groundTruthMTF = {"H" : [1,1,1,1,1,1,1,1],
#                     "V" : [1,1,1,1,1,1,1,1] }
#     evaluateMTFEstimations(dirInMtfResult, mtfDirection, groundTruthMTF, pathLogFile)