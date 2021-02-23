import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from glob import glob
import numpy as np
import tensorflow as tf
from model import CreateModel
from tqdm import tqdm
import pandas as pd
pd.options.display.max_colwidth = 100

from numba import jit
from shutil import copyfile



def imread(path):
    img = tf.keras.preprocessing.image.load_img(path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255
    img -= 0.5
    return img

def im2uint8(img):
    img += 0.5
    img *= 255
    img = np.clip(img,0,255)
    img = np.round(img)
    img = np.squeeze(img)
    return img




@jit(nopython=True)
def calcSAD(img_result,img_GT):

    absdiff = np.abs(img_result - img_GT)
    SAD = np.sum(absdiff,axis=1)

    maxSAD = np.max(SAD)
    avgSAD = np.mean(SAD)

    return maxSAD, avgSAD



strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    MODEL = CreateModel(nfs=[4,8,16,24,40])


png_input = "../egis_3dluts/DATASET/PNGs/3DLUT_256_in.png"
img_input = imread(png_input)
img_input = np.reshape(img_input,(-1,1,1,3))

for DIR in  os.listdir('./experiments'):
    print(DIR)
    if 'gamma(1.100)' in DIR:
        png_GT = "../egis_3dluts/DATASET/PNGs/3DLUT_256_train_out_gamma(1.100).png"
    elif 'hsv(0.014)' in DIR:
        png_GT = "../egis_3dluts/DATASET/PNGs/3DLUT_256_train_out_hsv(0.014).png"
    img_GT = imread(png_GT)
    img_GT = np.reshape(img_GT,(-1,1,1,3))
    img_GT = im2uint8(img_GT)

    CSV = f'./experiments/{DIR}/result.csv'
    if os.path.isfile(CSV):
        DF = pd.read_csv(CSV)
    else:
        DF = pd.DataFrame(columns=['hdf5', 'maxSAD', 'avgSAD'])


    hdf5s = sorted(glob(f'./experiments/{DIR}/HDF5/*.hdf5',recursive=True))[5000::3]
    hdf5s = [i for i in hdf5s if i not in DF['hdf5'].tolist()]
    for hdf5 in tqdm(hdf5s):
        MODEL.load_weights(hdf5)
        img_result = MODEL.predict(img_input, batch_size=256**3)
        img_result = im2uint8(img_result)
        
        # tf.keras.preprocessing.image.save_img(    'test.png', np.reshape(img_result,(256*16,256*16,3)))

        maxSAD, avgSAD = calcSAD(img_result,img_GT)
        DF = DF.append({'hdf5':hdf5, 'maxSAD':maxSAD, 'avgSAD':avgSAD},ignore_index=True)

    DF = DF.sort_values(by=['maxSAD', 'avgSAD'])
    DF.to_csv(CSV, index=False)

    print(DF['hdf5'].iloc[0])
    print(DF)

    
    copyfile(DF['hdf5'].iloc[0], f'./experiments/{DIR}/best.hdf5')