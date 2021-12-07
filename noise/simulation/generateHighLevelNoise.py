import cv2
import math
import os
import numpy as np
import random
import glob

# Configure the default parameters.
##################### Configuration ############################
params = {}

# Flags
params["applyPhotonNoise"] = False
params["applyDarkCurrent"] = False
params["darksignalCorrection"] = False
params["applySourceFollwerNoise"] = False
params["applyKtcNoise"] = False

# General
params["temperature"] = 293.15 # in K
params["exposureTime"] = 0.00014 # in s
params["overExposureFactor"] = 1.0
params["sensorPixelSize"] =  0.005 # in mm^2
params["INPUT_IMG_BIT_DEPTH"] = 8.0
params["OUTPUT_IMG_BIT_DEPTH"] = 8.0
params["sensorType"] = 'ccd'
NmaxIn = np.power(2.0, params["INPUT_IMG_BIT_DEPTH"])
NmaxOut = np.power(2.0, params["OUTPUT_IMG_BIT_DEPTH"])

# Photons to electrons
params["fullWellSize"] = 20000.0 # in e-

# Electrons to charge
params["darkSignalFoM"] = 1.0 # nA/cm^2
params["darkSignalFPNFactor"] = 0.04
params["senseNodeGain"] = 5.0 * 1e-6 # in V/e-
params["corrDoubleSampStoSTime"] = 1e-6 # in s

# Charge to voltage
params["sourceFollGain"] = 1.0 # in V/V
params["neighborCorrFactor"] = 0.0005
params["flickerNoiseCornerFreq"] = 1e6 # in Hz
params["corrDoubleSampTimeFact"] = 0.5
params["sourceFollowerCurrMod"] = 1e-8; # in A, CMOS only
params["fClock"] = 20 * 1e6 # in Hz
params["thermalWhiteNoise"] = 15 * 1e-9 # in V/(Hz)^0.5
params["senseNodeResetFactor"] = 1.0 # 1.0 = fully compensated by CDS; 0.0 = no compensation
params["sourceFollNonLinRatio"] = 1.05
params["corrDoubleSampGain"]      = 1.0 # in V/V; Correlated Double Sampling gain, lower means amplifying the noise.
randTelNoiseTimeConst = 0.1 * params["corrDoubleSampTimeFact"] * params["corrDoubleSampStoSTime"]; # in s

# Voltage to digital numbers
params["chargeNodeRefVolt"] = 3.1 # in V
params["adcNonLinRatio"] = 1.04

# Constants
siEGap0 = 1.1557 # in eV for silicon
siAlpha = 7.021*1e-4 # in eV/K for silicon
siBeta = 1108.0 # in K for for silicon
boltzmannConstEV = 8.617343 * 1e-5 # in eV/K
boltzmannConstJK = 1.3806504 * 1e-23 # in J/K
k1 = 1.0909e-14
q = 1.602176487 * 1e-19

senseNodeCap = q / params["senseNodeGain"]
Vmax = params["fullWellSize"] * q / senseNodeCap
#################################################################

def sampleRandGauss(mean, std):
    U, V, W = np.random.uniform(1e-10, 1.0, mean.shape), np.random.uniform(0.0, 1.0, mean.shape), np.random.uniform(0.0, 1.0, mean.shape)
    return np.where(
                W < 0.5, 
                (np.sqrt(-2.0 * np.log(U)) * np.sin(2.0 * math.pi * V)) * std + mean, 
                (np.sqrt(-2.0 * np.log(U)) * np.cos(2.0 * math.pi * V)) * std + mean
            )

def sampleRandPoisson(mean):
    return np.random.poisson(mean).astype("float64")

def estimatePhotonNumber(intensity, fullWell):
	oldRange = 1.0 - 0.0
	newRange = fullWell - 0.0
	return intensity * newRange / oldRange

def sourceFollPowerSpec(freq):
	return math.pow(params["thermalWhiteNoise"], 2.0) * (1.0 + params["flickerNoiseCornerFreq"] / freq) + randTelNoisePowerSpec(freq);


def randTelNoisePowerSpec(freq):
    if params["sensorType"].lower() == 'cmos':
    	termA = 2.0 * math.pow(params["sourceFollowerCurrMod"], 2.0) * randTelNoiseTimeConst;
    	termB = 4.0 + math.pow(2.0 * math.pi * freq * randTelNoiseTimeConst, 2.0);
    	return termA / termB;
    else:
        return 0.0

def corrDoubleSampTrans(freq):
	termA = 1.0 / (1.0 + math.pow(2.0 * math.pi * freq * params["corrDoubleSampTimeFact"] * params["corrDoubleSampStoSTime"], 2.0));
	termB = 2.0 - 2.0 * math.cos(2.0 * math.pi * freq * params["corrDoubleSampStoSTime"]);
	return termA * termB;
    
    
def logParams(dirOut, p):
    with open(os.path.join(dirOut,'parameters.txt'), 'w') as f:
        print(p, file=f)
        
def applyOverexposure(img, overExpFact):
    img = img * float(overExpFact)
    return img.clip(0.0, 1.0)
            
def applyHighLevelNoise(imgIn, noiseLevel, paramsIn={}, debug=True):
    """
    Simulate noise and add it to a given input image.
    :param imgIn: Input image with three dimensional shape (width, height, channels).
    :param noiseLevel: Desired mean noise level in digital numbers (e.g., between 1 and 30).
    :param paramsIn: Dictionary with configuration parameters overriding the default parameters.
    :param debug: Flag to print debug messages.
    :return: Noised input image.
    """
     
    # Check if image has three dimensions.
    if len(imgIn.shape) == 3:
        height, width, channels = imgIn.shape
    else:
        raise Exception("Expected input image to have 3 dimensions. Input image shape:", imgIn.shape)
         
     # Initialize uniform matrices.
    zeroes = np.full((height, width, channels), 0.0)
    ones = np.full((height, width, channels), 1.0)
    
    # Override default parameters
    if len(paramsIn) > 0:
       for key, val in paramsIn.items():
           params[key] = val
           if debug:
               print("Set parameter ", key, " to ", val)
    
    # Apply overexposure
    imgIn = applyOverexposure(imgIn, params["overExposureFactor"])
    
    # Convert photons to electrons
    photons = estimatePhotonNumber(imgIn, params["fullWellSize"])
    
    # Add photon noise
    electrons = photons;
    if(params["applyPhotonNoise"]):
    	    electrons = sampleRandPoisson(photons)
        
    # Add dark current and dark current shot noise
    temperature = params["temperature"] * ones
    if(params["applyDarkCurrent"]):
        eGap = siEGap0 - ((siAlpha * np.power(temperature, 2.0)) / (temperature + siBeta));
        dsTemp = np.power(temperature, 1.5) * np.exp(-1.0 * eGap / (2.0 * boltzmannConstEV * temperature));
        darkSignal =  2.55 * 1e15 * params["exposureTime"] * np.power(params["sensorPixelSize"] / 10.0, 2.0) * params["darkSignalFoM"] * dsTemp;
        darkSignalWithDarkNoise = sampleRandPoisson(darkSignal);
        if params["darksignalCorrection"]:
            darkSignalWithDarkNoise -= darkSignal
        electrons += darkSignalWithDarkNoise;
        
    # Source follower noise
    if(params["applySourceFollwerNoise"]):
        stepSize = 50000.0;
        f = 1.0;
        sfFreqSum = 0.0;
        while(f <= params["fClock"]):
            sfFreqSum += sourceFollPowerSpec(f) * corrDoubleSampTrans(f);
            f += stepSize;
    	
        sourceFollStdDev = np.sqrt(sfFreqSum) / (params["senseNodeGain"] * params["sourceFollGain"] * (1.0 - np.exp(-params["corrDoubleSampStoSTime"] / (params["corrDoubleSampTimeFact"] * params["corrDoubleSampStoSTime"]))));
        sourceFollStdDevMat = np.full((height, width, channels), sourceFollStdDev)
        electrons += np.round(sampleRandGauss(zeroes, sourceFollStdDevMat));
    
    # Truncate to full well size and round to full electrons
    electrons = np.floor(np.clip(electrons, 0.0, params["fullWellSize"]));
                
    # Convert charge to voltage
    chargeNodeRefVoltMat = np.full((height, width, channels), params["chargeNodeRefVolt"])
    if params["sensorType"].lower() == 'cmos':
        # Add kTc noise
        if (params["applyKtcNoise"]):
            ktcNoiseStdDev = np.sqrt((boltzmannConstJK * temperature) / senseNodeCap);
            ktcNoiseStdDevMat = np.full((height, width, channels), ktcNoiseStdDev)
            ktcNoise = np.exp(sampleRandGauss(zeroes, ktcNoiseStdDevMat)) - 1.0;
            voltage = (chargeNodeRefVoltMat + params["senseNodeResetFactor"] * ktcNoise) - (electrons * params["senseNodeGain"]);
        else:
            voltage = chargeNodeRefVoltMat - (electrons * params["senseNodeGain"]);
    elif params["sensorType"].lower() == 'ccd':
        voltage = chargeNodeRefVoltMat - (electrons * params["senseNodeGain"]);
    else:
        raise Exception("Unsupported sensor type:", params["sensorType"])
       
    # Apply source follower gain
    voltage *=  params["sourceFollGain"]
    
    # Add influence of correlated double sampling
    voltage *= params["corrDoubleSampGain"];
    
    # Convert voltage to digital numbers
    Vmin = q * params["senseNodeGain"] / senseNodeCap
    adGain = NmaxOut /(Vmax - Vmin)
    dn = adGain * (chargeNodeRefVoltMat - voltage)
    
    # Get simulated noise
    imgIn = imgIn * (NmaxOut - 1.0)
    noiseMap = dn - imgIn
    
    # Amplify noise to reach the desired noise level
    rawNoiseLevel = np.mean(np.abs(noiseMap))
    if debug:
        print("Raw Noise Level:", rawNoiseLevel)
    noiseAmpFactor = noiseLevel / rawNoiseLevel
    noiseMap = noiseAmpFactor * noiseMap
    
    # Add final noise to input image
    dn = imgIn + noiseMap
    dn = np.round(dn)
    dn = np.clip(dn, 0.0, NmaxOut - 1.0)
    
    return dn, noiseMap
 
def prepareImage(img):
    """
    Convert image to three dimensions (widht, height, channels) and normalize intensity values to range [0,1].
    :param img: Input gray-scale image with dimensions (widht, height) or (width, height, channels) with #(channels) in {1,3}.
    :return: Input image of dimension (width, height, channels=1) with intensities in range [0,1].
    """
    if len(img.shape) == 3 and img.shape[2] == 1:
        pass
    if len(img.shape) == 3 and img.shape[2] != 1:
        img = img[..., 1]
        img = np.expand_dims(img, axis=2)
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    else:
        raise Exception("Unexpected image dimension. Expected: (widht, height) or (width, height, channels) with #(channels) in {1,3}. Given:", img.shape)
    
    img = img.astype("float32") 
    if np.max(img) > 1.0:
        img = img / (NmaxIn - 1.0)
    
    return img

def applyPhotonNoise(img, noiseLevel):
    """
    Apply only photon noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Photon noised input image.
    """
    img = prepareImage(img)
    
    # Init parameters
    paramsIn = {}
    params["applyPhotonNoise"] = True
    params["applyDarkCurrent"] = False
    params["darksignalCorrection"] = False
    params["applySourceFollwerNoise"] = False
    params["applyKtcNoise"] = False
    
    img, noiseMap = applyHighLevelNoise(img, noiseLevel, paramsIn, debug=False)
    img = img.astype("uint8")
    img = img[..., 0]
    return img, noiseMap

def applyDarkCurrentShotNoise(img, noiseLevel):
    """
    Apply dark current shot noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Dark current shot noised input image.
    """
    img = prepareImage(img)
        
    # Init parameters
    paramsIn = {}
    paramsIn["exposureTime"] = 1.0/10.0
    paramsIn["temperature"] = 330.0
    paramsIn["sensorType"] = 'cmos'
    paramsIn["applyPhotonNoise"] = False
    paramsIn["applyDarkCurrent"] = True
    paramsIn["darksignalCorrection"] = True
    paramsIn["applySourceFollwerNoise"] = False
    paramsIn["applyKtcNoise"] = False
    
    img, noise = applyHighLevelNoise(img, noiseLevel, paramsIn, debug=False)
    img = img.astype("uint8")
    img = img[..., 0]
    return img, noise

def applyReadNoise(img, noiseLevel):
    """
    Apply read noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Read noised input image.
    """
    img = prepareImage(img)
        
    # Init parameters
    paramsIn = {}
    paramsIn["temperature"] = 330
    paramsIn["sensorType"] = 'cmos'
    paramsIn["applyPhotonNoise"] = False
    paramsIn["applyDarkCurrent"] = False
    paramsIn["darksignalCorrection"] = False
    paramsIn["applySourceFollwerNoise"] = True
    paramsIn["applyKtcNoise"] = True
    
    img, noise = applyHighLevelNoise(img, noiseLevel, paramsIn, debug=False)
    img = img.astype("uint8")
    img = img[..., 0]
    return img, noise

def applyPhotonDarkReadNoise(img, noiseLevel):
    """
    Apply photon noise, dark current shot noise and read noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Photon noised, dark current shot noised and read noised input image.
    """
    img = prepareImage(img)
        
    # Init parameters
    paramsIn = {}
    paramsIn["exposureTime"] = random.uniform(1.0/500.0, 1.0)
    paramsIn["temperature"] = random.uniform(300.0, 330.0)
    paramsIn["sensorType"] = 'cmos'
    paramsIn["applyPhotonNoise"] = True
    paramsIn["applyDarkCurrent"] = True
    paramsIn["darksignalCorrection"] = True
    paramsIn["applySourceFollwerNoise"] = True
    paramsIn["applyKtcNoise"] = True
    
    img, noise = applyHighLevelNoise(img, noiseLevel, paramsIn, debug=False)
    img = img.astype("uint8")
    img = img[..., 0]
    return img, noise

def applyDarkReadNoise(img, noiseLevel):
    """
    Apply dark current shot noise and read noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Dark current shot noised and read noised input image.
    """
    img = prepareImage(img)
        
    # Init parameters
    paramsIn = {}
    paramsIn["exposureTime"] = random.uniform(1.0/500.0, 1.0)
    paramsIn["temperature"] = random.uniform(300.0, 330.0)
    paramsIn["sensorType"] = 'cmos'
    paramsIn["applyPhotonNoise"] = False
    paramsIn["applyDarkCurrent"] = True
    paramsIn["darksignalCorrection"] = True
    paramsIn["applySourceFollwerNoise"] = True
    paramsIn["applyKtcNoise"] = True
    
    img, noise = applyHighLevelNoise(img, noiseLevel, paramsIn, debug=False)
    img = img.astype("uint8")
    img = img[..., 0]
    return img, noise


# Example for photon noise with noise level 25.
# if __name__ == "__main__":  
#     dirIn = r"..\..\data\udacity\img\GT"
#     dirOut = r"..\..\data\udacity\img\noised"
#     imgFileEnding = ".jpg"
#     noiseLevel = 25

#     subDirOut = os.path.join(dirOut, str(noiseLevel))
#     if not os.path.exists(subDirOut):
#         os.makedirs(subDirOut)

#     imgPaths = glob.glob(os.path.join(dirIn, "*" + imgFileEnding))
#     for imgPath in imgPaths:
#         img = cv2.imread(imgPath).astype("float32") / (NmaxIn - 1.0)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img, _ = applyPhotonNoise(img, noiseLevel)
#         imgName = imgPath.split(os.sep)[-1].split(".")[0]
#         cv2.imwrite(os.path.join(subDirOut, imgName + imgFileEnding), img)    