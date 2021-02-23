from argparse import ArgumentParser, RawTextHelpFormatter
# =================================================
# ArgumentParser
# =================================================
parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-ki", default=3,type=int)
parser.add_argument("-gpu", default='0',type=str)
parser.add_argument("-opt", default=0,type=int)
parser.add_argument("-nfs", default='4,8,16,24',type=str)
args = parser.parse_args()
ki = args.ki
gpu = args.gpu
opt = args.opt
nfs = eval(f'[{args.nfs}]')
kernel_initializer = ['RandomNormal', 'RandomUniform', 'glorot_normal', 'glorot_uniform', 'lecun_uniform', 'lecun_normal']
Optimizers = ['Adamax', 'SGD']

print(f'kernel_initializer:{kernel_initializer[ki]}')
print(f'gpu:{gpu}')
print(f'opt:{Optimizers[opt]}')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['PYTHONHASHSEED'] = '0'


from datetime import datetime
import numpy as np
from tqdm import tqdm

from model import CreateModel, maxSAD, avgSAD, myAcc
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay



NOW = datetime.now().strftime("%Y%m%d_%H%M%S")


SUB_EPOCH = 10000



def getData(path,N=None):
    with open(path) as f:
        lines = f.readlines()

    N = len(lines) if N==None else N

    X = np.zeros((N,1,1,3))
    Y = np.zeros((N,1,1,3))
    for idx, line in enumerate(tqdm(lines[:N])):
        pix1, pix2 = line.split(';')

        pix1 = eval(f'np.array([[[[{pix1}]]]])')
        pix2 = eval(f'np.array([[[[{pix2}]]]])')

        X[idx,:,:,:] = (pix1/255)-0.5
        Y[idx,:,:,:] = (pix2/255)-0.5
    return X, Y

def val_result(x, y_true, y_pred):
    yt = np.round(255*(y_true+0.5))
    yp = np.round(255*(y_pred+0.5))
    yp = np.clip(yp,0,255)
    absdiff = np.abs(yt-yp)

    SAD = np.sum(absdiff, axis=3)
    maxSAD = np.max(SAD)
    avgSAD = np.mean(SAD)

    HARD = np.argwhere(np.squeeze(SAD) > 0)
    N = len(HARD)
    
    x_hard = np.zeros((N,1,1,3))
    y_hard = np.zeros((N,1,1,3))
    for idx, hardidx in enumerate(tqdm(HARD)):
        x_hard[idx,:,:,:] = x[hardidx,:,:,:]
        y_hard[idx,:,:,:] = y_true[hardidx,:,:,:]

    return maxSAD, avgSAD, x_hard, y_hard










# if os.path.isfile('./DATA/gamma(1.100)_256.npz'):
#     npzfile = np.load('./DATA/gamma(1.100)_256.npz')
#     X = npzfile['arr_0']
#     Y = npzfile['arr_1']
#     del npzfile
# else:
#     X, Y = getData("/home/kobe_nein/egis_3dluts/DATASET/TXT_YML/gamma(1.100)_256.txt")
#     np.savez('gamma(1.100)_256.npz', X, Y)



if os.path.isfile('./DATA/hsv(0.014)_256.npz'):
    npzfile = np.load('./DATA/hsv(0.014)_256.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    del npzfile
else:
    X, Y = getData("/home/kobe_nein/egis_3dluts/DATASET/TXT_YML/hsv(0.014)_256.txt")
    np.savez('./DATAhsv(0.014)_256.npz', X, Y)




# =============================================================================
if Optimizers[opt] == 'SGD':
    optimizer = SGD(learning_rate=tf.keras.experimental.CosineDecayRestarts(1e-2, 100))
    # optimizer = SGD(learning_rate=ExponentialDecay(1e-2, 64, 0.8))
elif Optimizers[opt] == 'Adamax':
    # optimizer = tf.keras.optimizers.Adamax(learning_rate=0.00001)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=tf.keras.experimental.CosineDecayRestarts(0.001, 100))



HDF5 = f'weights_{kernel_initializer[ki]}_{Optimizers[opt]}.hdf5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=HDF5, monitor='maxSAD', mode='min', verbose=0, save_best_only=True, save_weights_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(f'./Logs/{NOW}_{kernel_initializer[ki]}_{Optimizers[opt]}.csv')
my_callbacks = [
    model_checkpoint,
    csv_logger,
    # tf.keras.callbacks.EarlyStopping(patience=2),
    # tf.keras.callbacks.TensorBoard(log_dir=f'./Logs/{NOW}_{kernel_initializer[ki]}_{Optimizers[opt]}'),
]
# =============================================================================
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    MODEL = CreateModel(KI=kernel_initializer[ki], nfs=nfs)
    MODEL.compile( loss=MeanSquaredError(), optimizer=optimizer, metrics=[myAcc, maxSAD, avgSAD])

if os.path.isfile(HDF5):
    print(f"restore from {HDF5}")
    MODEL.load_weights(HDF5)

MODEL.evaluate(X, Y, batch_size=256**3)
MODEL.fit(X, Y, epochs=SUB_EPOCH, batch_size=256**2, callbacks=my_callbacks)

MODEL.evaluate(X, Y, batch_size=256**3)
MODEL.load_weights(HDF5)
MODEL.evaluate(X, Y, batch_size=256**3)

