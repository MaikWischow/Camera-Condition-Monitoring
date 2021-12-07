import sys
import glob
import os
import time
import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d
from generateDateset import DatabaseCreator

sys.path.append("../../Simulation")
from generateHighLevelNoise import applyPhotonDarkReadNoise

DEBUG = False
PATCH_SHAPE = [128,128,3]
MAX_IMG_INTENSITY = 255.0
IMG_FILE_ENDING = ".jpg"

def oneChannel2ThreeChannel(img):
    """
    Convert 1-Channel image to 3-Channel image.
    :param img: 1-Channel image.
    :return: 3-Channel image
    """
    return cv2.merge((img, img, img))

def rgb2gray(img):
    """
    Convert rgb to gray-scale image.
    :param img: RGB image in range [0,1]
    :return: 3-Channel gray-scale image in range [0,1]
    """
    img8bit = (img * MAX_IMG_INTENSITY).astype("uint8")
    img8bit = cv2.cvtColor(img8bit, cv2.COLOR_RGB2GRAY)
    img = oneChannel2ThreeChannel(img8bit)
    return (img.astype("float32") / MAX_IMG_INTENSITY)

def addNoise(img, generatedNoise):
    """
    Add noise to image and ensure intensity limits.
    :param img: 3-Channel image in range [0,1].
    :param generatedNoise: 1-Channel noise map in range [0,1] of the same size as img. 
    :return: 3-Channel noised image in range [0,1]
    """
    img = img + oneChannel2ThreeChannel(generatedNoise)
    img = np.clip(img, 0.0, 1.0)
    return img

def generateNoise(img, noiseLevel, ratio=0.5):
    """
    Add noise to image and ensure intensity limits.
    :param img: 3-Channel image in range [0,1].
    :param generatedNoise: 1-Channel noise map in range [0,1] of the same size as img. 
    :return: 3-Channel noised image in range [0,1]
    """
    noisedImg, _ = applyPhotonDarkReadNoise(img, noiseLevel)
    return noisedImg

class Estimator:
    """
    A class to train and test a tensorflow estimator.
    """

    # predict_op = []
    def __init__(self, batchSize = 1, depth = 8, feature_dim = 8, device = '/gpu:0', xshape=PATCH_SHAPE, yshape=PATCH_SHAPE, lr=1e-5):
        self.batchSize = batchSize
        self.depth = depth
        self.feature_dim = feature_dim
        self.device = device
        self.xshape = xshape
        self.yshape = yshape
        self.lr = lr

    def init_weights(self, shape, name):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def residual_block(self, h, width, kernel_size, depth):
        h_in = h
        for i in range(depth):
            h = conv2d(h, width, kernel_size)
        return h_in + h

    def build_model(self, bTrain):
        assert len(self.xshape) == 3
        # place holders
        x = tf.placeholder('float', [self.batchSize, self.xshape[0], self.xshape[1], self.xshape[2]], 'x')

        if bTrain:
            noise_level = tf.placeholder('float', shape=(1), name='noise') # in [0, 1]
            noise = tf.fill([self.batchSize, self.xshape[0], self.xshape[1], 1], noise_level[0])
            h = x

        else:
            h = x

        # start data flow
        block_depth = 4
        num_block = self.depth // block_depth
        for d in range(0, num_block):
            h = conv2d(h, self.feature_dim, [3, 3])
            h = self.residual_block(h, self.feature_dim, [3, 3], block_depth)

        h = conv2d(h, 1 , [3, 3])

        y_conv = h
        scalar_en = tf.reduce_mean(h)

        # loss function
        if bTrain:
            lmd = 0.25
            
            cost_mat = tf.reduce_sum(tf.square(tf.subtract(noise,  y_conv))) / self.batchSize
            cost_scalar = tf.square(tf.subtract(scalar_en, noise_level[0]))

            cost = lmd * cost_mat + (1 - lmd) * cost_scalar
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)
            
            return y_conv, train_op, cost, x, noise_level
        else:
            return y_conv, x


    def train(self, saveDir,  trY,  valY, minNoiseLevel, maxNoiseLevel, maxEpoch=100, part=0):
        """
        Train the model.
        :param trX:
        :param trY:
        :param maxEpoch:
        :param batchSize:
        :return:
        """

        # add new axis for data
        if trY.ndim == 3:
            trY = trY[..., np.newaxis]
        if valY.ndim == 3:
            valY = valY[..., np.newaxis]

        # generate model
        if not hasattr(self, 'predict_op'):
            print('Building model ...')
            self.predict_op, self.train_op, self.cost, self.x, self.noise_level = self.build_model(bTrain=True)
        # Launch the graph in a session
        saver = tf.train.Saver()

        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)

        curEpoch = 0
        bestLoss = 10e6
        if os.path.isfile(saveDir + '/loss.txt'):
            with open(saveDir + '/loss.txt', 'r') as log_file:
                log = log_file.readlines()
                if len(log) > 0:
                    curEpoch = int(log[-1].split(' ')[0]) + 1 + part * maxEpoch

        out_file = open(saveDir + '/loss.txt', 'a')
        with tf.Session() as sess:
            self.sess = sess

            with tf.device(self.device):
                ckpt = tf.train.get_checkpoint_state(saveDir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('Restored training...')
                    saver.restore(sess, saveDir + '/tf_estimator.ckpt')
                else:
                    print('Start training...')
                    # init all variables
                    tf.global_variables_initializer().run()

                for i in range(int(curEpoch), int(maxEpoch)):
                    start_time = time.time()
                    np.random.shuffle(trY)
                    
                    print('Epoch %d ...' % i)
                    for start, end in zip(range(0, len(trY), self.batchSize),
                                          range(self.batchSize, len(trY) + 1, self.batchSize)):

                        # Get next image and convert it to gray-scale
                        y = trY[start:end][0, ...]
                        y = rgb2gray(y)
                        
                        # Scale the noise level limits to range [0,1] and generate a noise level
                        minNoiseLevel /= MAX_IMG_INTENSITY
                        maxNoiseLevel /= MAX_IMG_INTENSITY
                        noiseLevel = np.random.rand(1) * (maxNoiseLevel - minNoiseLevel) + minNoiseLevel
                        noiseLevelInt8 = noiseLevel * MAX_IMG_INTENSITY
                        
                        # Generate noise map and convert it to the same range and #channels as the input image
                        y = generateNoise(y, noiseLevelInt8)
                        y = y.astype("float32") / MAX_IMG_INTENSITY
                        y = oneChannel2ThreeChannel(y)    
                        y = y[np.newaxis, ...]
                        
                        # Run the Neural Network (NN) on the image and the noise level
                        sess.run(self.train_op, feed_dict={self.x: y, self.noise_level: noiseLevel})

                    # Print loss for selected noise levels.
                    for n_levelInt8 in [1, 10, 15, 30]:
                        n_level_ = n_levelInt8 / MAX_IMG_INTENSITY
                        randTrYIndex = np.random.randint(0, len(trY) - self.batchSize)
                        randValYIndex = np.random.randint(0, len(valY) - self.batchSize)
                        
                        # Get a random image from the validation sub-dataset, apply random noise
                        # and prepare it for the NN.
                        yVal = valY[randValYIndex : randValYIndex + self.batchSize, ...][0, ...]
                        yVal = rgb2gray(yVal)
                        yVal = generateNoise(yVal, n_levelInt8)
                        yVal = yVal.astype("float32") / MAX_IMG_INTENSITY
                        yVal = oneChannel2ThreeChannel(yVal)  
                        yVal = yVal[np.newaxis, ...]
                        
                        # For comparison: get a random image from the training sub-dataset, apply random noise
                        # and prepare it for the NN.
                        yTr = trY[randTrYIndex : randTrYIndex + self.batchSize, ...][0, ...]
                        yTr = rgb2gray(yTr)
                        yTr = generateNoise(yTr, n_levelInt8)
                        yTr = yTr.astype("float32") / MAX_IMG_INTENSITY
                        yTr = oneChannel2ThreeChannel(yTr)  
                        yTr = yTr[np.newaxis, ...]
                        
                        # Apply both images to the NN and print losses
                        loss = sess.run(self.cost, feed_dict={self.x: yTr, self.noise_level: [n_level_]})
                        val_loss = sess.run(self.cost, feed_dict={self.x: yVal, self.noise_level: [n_level_]})
                        print('loss n : ', n_levelInt8, loss, ' val loss : ', val_loss)
                        print(i, n_levelInt8, loss, val_loss, file=out_file)
                        
                    print('time : ', time.time() - start_time, ' s')

                    # Save checkpoint every second epoch, if loss is best or NN is near end of training.
                    if i % 2 == 0:
                        if val_loss < bestLoss or i < maxEpoch * 4 / 5:
                            bestLoss = val_loss
                            saver.save(sess, saveDir + '/tf_estimator.ckpt')
                            print('Model saved')
                            print('Best Loss ', bestLoss)
                        out_file.flush()

                    if i > maxEpoch * 4 / 5 and val_loss < bestLoss:
                        bestLoss = val_loss
                        saver.save(sess, saveDir + '/tf_estimator.ckpt')
                        print('Model saved')
                        print('Best Loss ', bestLoss)

        out_file.close()
        print('Best Loss ', bestLoss)


    def load_model(self, saveDir, batchSize=1, xshape=PATCH_SHAPE, yshape=PATCH_SHAPE):
        """
        Load a model from a checkpoint.
        """
        self.batchSize = batchSize
        self.xshape = xshape
        self.yshape = yshape
        self.predict_op, self.x = self.build_model(bTrain=False)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        with tf.device(self.device):
            ckpt = tf.train.get_checkpoint_state(saveDir)
            if ckpt and ckpt.model_checkpoint_path:
                print('loading model ...')
                # vars_list = tf.train.list_variables(saveDir)
                saver.restore(sess, saveDir + '/tf_estimator.ckpt')
                self.sess = sess

    def estimateNoise(self, image, psize, crop):
        """
        Estimate noise of an image patch-wise.
        :param image: noised image
        :param psize: size of patch
        :param crop:  crop of image patch
        :return:
        """
        assert image.ndim == 3
        start_time = time.time()

        h, w = image.shape[:2]

        psize = min(min(psize, h), w)
        psize -= psize % 2

        patch_step = psize
        patch_step -= 2 * crop
        shift_factor = 2

        # Result array
        R = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)

        rangex = range(0, w - 2 * crop, patch_step)
        rangey = range(0, h - 2 * crop, patch_step)
        ntiles = len(rangex) * len(rangey)

        # resize input
        sess = self.sess
        with tf.device(self.device):
            with tqdm.tqdm(total=ntiles, unit='tiles', unit_scale=True) as pbar:
                for start_x in rangex:
                    for start_y in rangey:
                        a_time = time.time()

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

                        tileM = image[np.newaxis, start_y:end_y, start_x:end_x, :]
                        if DEBUG:
                            print('tileM.shape : ', tileM.shape)

                        b_time = time.time()
                        out = sess.run(self.predict_op, feed_dict={self.x: tileM })
                        c_time = time.time()

                        out = out.reshape(out.shape[1], out.shape[2], 1)
                        s = out.shape[0]
                        R[start_y + crop:start_y + crop + s,
                        start_x + crop:start_x + crop + s, :] = out

                        d_time = time.time()

                        pbar.update(1)

                        if DEBUG:
                            print('image crop : ', (b_time - a_time) * 1000, ' ms')
                            print('forward : ', (c_time - b_time) * 1000, ' ms')
                            print('put patch back :', (d_time - c_time) * 1000, ' ms')

        R[R < 0] = 0.0
        R[R > 1] = 1.0

        runtime = (time.time() - start_time) * 1000  # in ms

        return R, runtime


# #######################################################
# # Functions to call Estimator

def mem_divide(x, divider):
    # a memory efficient divide function
    # when x is huge, this method saves memory

    for i in range(0, x.shape[0]):
        x[i,...] = x[i, ...] / divider
    return x


def train(modelPath, trainPath, valPath, minNoiseLevel, maxNoiseLevel, feature_dim=64, depth=12, x_shape=PATCH_SHAPE, y_shape=PATCH_SHAPE, device='0'):
    """
    Training using Estimator class.
    :param modelPath: path to save trained model
    :param trainPath: path to training dataset
    :param valPath: path to validation dataset
    :param feature_dim: width of the DNN
    :param depth: depth of the DNN
    :param minNoiseLevel: minimum noise level added to clean images
    :param maxNoiseLevel: maximum noise level added to clean images
    :param x_shape: Input patch size
    :param y_shape: Output patch size
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :return: Null
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    estimator = Estimator(device='/gpu:0', depth= depth, feature_dim=feature_dim, xshape=x_shape, yshape=y_shape)

    dc = DatabaseCreator()

    name = 'rgb'
    maxEpoch = 100

    valY = dc.load_hdf5_v1(valPath, name)
    valY = mem_divide(valY, MAX_IMG_INTENSITY)
    npart = dc.load_hdf5_v1(trainPath, 'npart')

    curEpoch = 0
    if os.path.isfile(modelPath + '/loss.txt'):
        with open(modelPath + '/loss.txt', 'r') as log_file:
            log = log_file.readlines()
            if len(log) > 0:
                curEpoch = int(log[-1].split(' ')[0])
                
    trY = dc.load_hdf5_v1(trainPath, name)
    trY = mem_divide(trY, MAX_IMG_INTENSITY)

    for i in range(int((curEpoch+1) / (maxEpoch/npart)), npart):
        estimator.train(modelPath,  trY,  valY, minNoiseLevel, maxNoiseLevel, maxEpoch=maxEpoch / npart * (i+1))

    estimator.sess.close()

def prepareNoiseEstimationModel(modelPath, patchShape=[128, 128, 3]):
    """
    Creates the CNN model and load the given weights.
    :param modelPath: path to the saved weights file
    :param patchShape: Size for input image patches.
    :return: Null
    """
    estimator = Estimator(batchSize=1, feature_dim=64, depth=12, xshape=patchShape, yshape=patchShape)
    estimator.load_model(modelPath, batchSize=1, xshape = patchShape, yshape = patchShape)
    return estimator

def estimateNoise(estimator, imgPath, dirOut=None, saveResults=True):
    """
    Estimate image noise.
    :param estimator: prepared CNN estimator.
    :param imgPath: path to the image from which the noise is to be estimated.
    :param saveResults: decide whether to save the noise estimations.
    :param dirOut: output directory for saving noise estimations.
    :return: Null
    """
    try:
        # Load and convert image to range [0,1]
        img = np.array(cv2.imread(imgPath))
        img = img / MAX_IMG_INTENSITY
    
        # Estimate image noise
        startTime = time.time()
        estimatedNoiseMap, _ = estimator.estimateNoise(img, PATCH_SHAPE[0], 0)
        endTime = time.time()
        print("Time per image:", endTime - startTime, "s.")
        
        # Convert noise estimation to range [0, 255]
        estimatedNoiseMap *= MAX_IMG_INTENSITY
        estimatedNoiseMap = estimatedNoiseMap.astype("uint8")
        
        # Save the result as .npz file (but do not override existing files)
        if saveResults:
            if dirOut is not None:
                if not os.path.exists(dirOut):
                    os.makedirs(dirOut)
                    
                imgName = imgPath.split(os.sep)[-1].split(".")[0]
                noiseMapPath = os.path.join(dirOut, imgName + ".npz")
                if not os.path.exists(noiseMapPath):
                    np.savez_compressed(noiseMapPath, estimatedNoiseMap[..., 0])
                else:
                    print("The noise estimation result file", noiseMapPath, "already exists. Did not override the result.")
        
        return estimatedNoiseMap
    except:
        return None


# Example
# if __name__ == '__main__':
#     tf.reset_default_graph()
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
#     # Train the model
#     if False:
#         modelPath = "models/20210517_photonDarkReadNoise_noiseLvl0-30_test"
#         trainDatasetPath = r"./train.h5" # Not included in this repository.
#         testDatasetPath = r"./test.h5" # Not included in this repository.
#         minNoiseLevel = 0.0
#         maxNoiseLevel = 30.0
#         train(modelPath, trainDatasetPath, testDatasetPath, minNoiseLevel, maxNoiseLevel)
       
#     # Test the model
#     if False:
#         subDir = "GT"
#         dirIn = r"../../../data/udacity/img"
#         dirOut = r"../../../data/udacity/labels_noise_patchwise/CNN"

#         subDirOut = os.path.join(dirOut, subDir)
#         if not os.path.exists(subDirOut):
#            os.makedirs(subDirOut)

#         modelPath = "models/20210517_photonDarkReadNoise_noiseLvl0-30"
#         estimator = prepareNoiseEstimationModel(modelPath)
#         for imgPath in glob.glob(os.path.join(dirIn, subDir, "*" + IMG_FILE_ENDING)):
#             noiseMap = estimateNoise(estimator, imgPath, subDirOut, True)
#             print("Median noise level:", np.median(noiseMap))