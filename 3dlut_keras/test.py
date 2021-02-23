import numpy as np
import tensorflow as tf
from model import CreateModel



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


def CalcSAD(input_png_path, GT_png_path):
    img_input = imread(input_png_path)
    img_GT = imread(GT_png_path)

    # reshape for loop
    img_input = np.reshape(img_input,(-1,1,1,3))
    img_GT = np.reshape(img_GT,(-1,1,1,3))


    MODEL = CreateModel()
    MODEL.load_weights('weights_lecun_normal_Adamax.hdf5')
    img_result = MODEL.predict(img_input, batch_size=256**3)

    img_result = im2uint8(img_result)
    img_GT = im2uint8(img_GT)
    absdiff = np.abs(img_result - img_GT)
    SAD = np.sum(absdiff,axis=1)

    maxSAD = np.max(SAD)
    avgSAD = np.mean(SAD)

    return maxSAD, avgSAD



png_input = "../egis_3dluts/DATASET/PNGs/3DLUT_256_in.png"
png_GT = "../egis_3dluts/DATASET/PNGs/3DLUT_256_train_out_gamma(1.100).png"

maxSAD, avgSAD = CalcSAD(png_input, png_GT)

print(maxSAD, avgSAD)