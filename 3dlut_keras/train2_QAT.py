from argparse import ArgumentParser, RawTextHelpFormatter
# =================================================
# ArgumentParser
# =================================================
parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-gpu", default='0', type=str)
parser.add_argument("-json", type=str)
args = parser.parse_args()

print(f'gpu:{args.gpu}')
print(f'json:{args.json}')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['PYTHONHASHSEED'] = '0'


from datetime import datetime
import numpy as np
# from tqdm import tqdm
from config import getConf

from model_QAT import CreateModel, maxSAD, avgSAD, myAcc
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard



conf = getConf(args.json)
for k,v in conf.items():
    if isinstance(v,dict):
        print(f"[{k}]")
        for kk,vv in v.items():
            print(f"    {kk}:{vv}")
    else:
        print(f"{k}:{v}")


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
if os.path.isdir(f"./experiments/{conf['name']}"):
    os.mkdir(f"./experiments/{conf['name']}_{NOW}")
    os.system(f"mv ./experiments/{conf['name']}/* ./experiments/{conf['name']}_{NOW}/")
else:
    os.mkdir(f"./experiments/{conf['name']}")

hdf5_dir = f"./experiments/{conf['name']}/HDF5"
os.mkdir(hdf5_dir)

# schedules
if conf['parameter']['schedules'] == 'CosineDecayRestarts':
    from tensorflow.keras.experimental import CosineDecayRestarts
    learning_rate = CosineDecayRestarts(conf['parameter']['learning_rate'], 100)
elif conf['parameter']['schedules'] == 'ExponentialDecay':
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    learning_rate = ExponentialDecay(conf['parameter']['learning_rate'])
else:
    learning_rate = conf['parameter']['learning_rate']
# =============================================================================

# optimizer
if conf['parameter']['Optimizers'] == 'SGD':
    from tensorflow.keras.optimizers import SGD
    optimizer = SGD(learning_rate)
else:
    from tensorflow.keras.optimizers import Adamax
    optimizer = Adamax(learning_rate)
# =============================================================================

# callbacks
callbacks = [
    ModelCheckpoint(filepath=os.path.join(hdf5_dir,'weights{epoch:08d}.hdf5'), monitor='maxSAD', mode='min', verbose=0,  save_weights_only=True, save_best_only=False),
    CSVLogger(f"./experiments/{conf['name']}/log.csv"),
    TensorBoard(f"./experiments/{conf['name']}/Logs"),
    # tf.keras.callbacks.EarlyStopping(patience=2),
]
# =============================================================================

MODEL = CreateModel(KI=conf['parameter']['kernel_initializer'], nfs=conf['nfs'])
MODEL = tfmot.quantization.keras.quantize_model(MODEL)
MODEL.compile( loss=MeanSquaredError(), optimizer=optimizer, metrics=[myAcc, maxSAD, avgSAD])

# if conf['restore_from']:
#     print(f"restore from conf['restore_from']")
#     MODEL.load_weights(conf['restore_from'])

npzfile = np.load(conf['dataset']['XY_npz'])
X = (npzfile['arr_0']+0.5)*255 - 128
X = X.astype(np.int8)
Y = (npzfile['arr_1']+0.5)*255 - 128
Y = Y.astype(np.int8)
del npzfile

print(X)

# MODEL.evaluate(X, Y, batch_size=256**3)
MODEL.fit(X, Y, epochs=conf['parameter']['epoch'], batch_size=256**2, callbacks=callbacks)
# MODEL.evaluate(X, Y, batch_size=256**3)
