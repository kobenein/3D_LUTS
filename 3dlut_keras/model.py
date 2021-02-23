import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Add
from tensorflow.keras.callbacks import Callback

from keras_flops import get_flops


def ResidualBlock_noBN(blockInput, nf, res=True, KI="glorot_uniform"):
    x = Conv2D(nf, 1, 1, kernel_initializer=KI, activation='relu')(blockInput)
    x = Conv2D(nf, 1, 1, kernel_initializer=KI)(x)
    if res:
        x = Add()([x, blockInput])
    return x

def CreateModel(res=True, nfs=[4,8,16,24], KI="glorot_uniform"):
    input_pix = tf.keras.Input(shape=(1, 1, 3))
    x = Conv2D(nfs[0], (1, 1), kernel_initializer=KI, activation=LeakyReLU(alpha=0.1))(input_pix)

    for nf in nfs[1:]:
        # print(nf)
        x = Conv2D(nf, (1, 1), kernel_initializer=KI, activation=LeakyReLU(alpha=0.1))(x)
        x = ResidualBlock_noBN(x, nf, res, KI=KI)
        x = ResidualBlock_noBN(x, nf, res, KI=KI)

    x = Conv2D(nfs[-1], (1, 1), kernel_initializer=KI, activation=LeakyReLU(alpha=0.1))(x)
    x = ResidualBlock_noBN(x, nfs[-1], res, KI=KI)
    x = ResidualBlock_noBN(x, nfs[-1], res, KI=KI)

    output_pix = Conv2D(3, (1, 1), kernel_initializer=KI, activation=LeakyReLU(alpha=0.1))(x)

    # output_pix = tf.keras.layers.Lambda( lambda x: tf.clip_by_value(x, -0.5, 0.5))(output_pix)
    
    model = tf.keras.Model(input_pix, output_pix, name="3D_LUT_Functional")
    

    # model.summary()
    # flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops} ({flops/10**9:.03}G)")
    # tf.keras.utils.plot_model(model, "3D_LUT.png", show_shapes=True, dpi=150)

    return model

def maxSAD(y_true, y_pred):
    y_true = tf.clip_by_value(tf.round(255*(y_true+0.5)), 0, 255)
    y_pred = tf.clip_by_value(tf.round(255*(y_pred+0.5)), 0, 255)
    absdiff = tf.abs(y_true-y_pred)
    SAD = tf.reduce_sum(absdiff, 3)
    maxSAD_ = tf.reduce_max(SAD)
    return maxSAD_

def avgSAD(y_true, y_pred):
    y_true = tf.clip_by_value(tf.round(255*(y_true+0.5)), 0, 255)
    y_pred = tf.clip_by_value(tf.round(255*(y_pred+0.5)), 0, 255)
    absdiff = tf.abs(y_true-y_pred)
    SAD = tf.reduce_sum(absdiff, 3)
    avgSAD_ = tf.reduce_mean(SAD)
    return avgSAD_

def myAcc(y_true, y_pred):
    y_true = tf.clip_by_value(tf.round(255*(y_true+0.5)), 0, 255)
    y_pred = tf.clip_by_value(tf.round(255*(y_pred+0.5)), 0, 255)
    absdiff = tf.abs(y_true-y_pred)
    SAD = tf.reduce_sum(absdiff, 3)
    return 1-tf.math.count_nonzero(SAD) / absdiff.shape[0]


if __name__ == '__main__':
    import numpy as np

    npzfile = np.load('./DATA/gamma(1.100)_256.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    del npzfile

    # =============================================================================
    optimizer = tf.keras.optimizers.SGD(learning_rate=tf.keras.experimental.CosineDecayRestarts(1e-2, 1000))
    # optimizer = SGD(learning_rate=ExponentialDecay(1e-2, 64*3, 0.9))
    # =============================================================================
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        MODEL = CreateModel()
        MODEL.compile( loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=[myAcc, maxSAD, avgSAD])
    HDF5 = 'weights.hdf5'
    MODEL.load_weights(HDF5)
    MODEL.fit(X, Y, epochs=1, batch_size=256**2, callbacks=[CustomCallback()])
    MODEL.evaluate(X, Y, batch_size=256**3, callbacks=[CustomCallback()])
    # =============================================================================

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(256**2)
