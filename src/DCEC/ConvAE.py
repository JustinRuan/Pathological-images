from keras import Input
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Concatenate, Add
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
from se import squeeze_excite_block

###
# 设计自编码器网络
###
def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    ### 普通的三层卷积的自编码器
    # model = Sequential()
    # model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    # model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
    # model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    # model.add(Flatten())
    # model.add(Dense(units=filters[3], name='embedding'))
    # model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))
    # model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    # model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    # model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    # model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    # model.summary()
    ### 普通的三层卷积的自编码器
    # inputs = Input(shape=[input_shape[0],input_shape[1],input_shape[2]],name='conv1_input')
    # x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    # x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    # x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    # x = Flatten()(x)
    # x = Dense(units=filters[3], name='embedding')(x)
    # x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(x)
    # x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    # x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    # x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    # x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    # model = Model(inputs=inputs, outputs=x)
    # model.summary()

### 在普通的三层卷积的自编码器中将其中一层替换为inception结构
    # inputs = Input(shape=[input_shape[0], input_shape[1], input_shape[2]], name='conv1_input')
    # x_1 = Conv2D(filters[0], 1, strides=2, padding='same', activation='relu', name='conv1_1')(inputs)
    # x_3 = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='conv1_2')(inputs)
    # x_5 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_3')(inputs)
    # x = Concatenate()([x_1, x_3, x_5])
    # x = Conv2D(filters[1]*2, 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    # x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    # x = Flatten()(x)
    # x = Dense(units=filters[3], name='embedding')(x)
    # x = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(x)
    # x = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(x)
    # x = Conv2DTranspose(filters[1]*2, 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    # x = Conv2DTranspose(filters[0]*3, 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    # x_1 = Conv2DTranspose(input_shape[2], 1, strides=2, padding='same', activation='relu', name='deconv1_1')(x)
    # x_3 = Conv2DTranspose(input_shape[2], 3, strides=2, padding='same', activation='relu', name='deconv1_2')(x)
    # x_5 = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', activation='relu', name='deconv1_3')(x)
    # x = Add()([x_1, x_3, x_5])
    # # x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    # model = Model(inputs=inputs, outputs=x)
    # model.summary()

### 在普通的三层卷积的自编码器中将添加一层inception结构，由于使用cifar10数据集，将网络提取的特征数变大
    # inputs = Input(shape=[input_shape[0], input_shape[1], input_shape[2]], name='conv1_input')
    # x = Conv2D(96, 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    # x_1 = Conv2D(64, 1, strides=2, padding='same', activation='relu', name='conv1_1')(x)
    # x_3 = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv1_2')(x)
    # x_5 = Conv2D(64, 5, strides=2, padding='same', activation='relu', name='conv1_3')(x)
    # x = Concatenate()([x_1, x_3, x_5])
    # x = Conv2D(192, 3, strides=2, padding='same', activation='relu', name='conv2')(x)
    # x = Conv2D(192, 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    # x = Flatten(name='embedding')(x)
    # # x = Dense(512 , name='embedding')(x)
    # # x = Dense(units=filters[2] * int(input_shape[0] / 16) * int(input_shape[0] / 16), activation='relu')(x)
    # x = Reshape((int(input_shape[0] / 16), int(input_shape[0] / 16), 192  ))(x)
    # x = Conv2DTranspose(192, 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    # x = Conv2DTranspose(192, 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
    # x_1 = Conv2DTranspose(32, 1, strides=2, padding='same', activation='relu', name='deconv1_1')(x)
    # x_3 = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu', name='deconv1_2')(x)
    # x_5 = Conv2DTranspose(32 , 5, strides=2, padding='same', activation='relu', name='deconv1_3')(x)
    # x = Concatenate()([x_1, x_3, x_5])
    # x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    # model = Model(inputs=inputs, outputs=x)
    # model.summary()

### 在普通的三层卷积的自编码器中添加se-block
    # inputs = Input(shape=[input_shape[0],input_shape[1],input_shape[2]],name='conv1_input')
    # x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    # x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    # x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    # x = squeeze_excite_block(x)
    # x = Flatten()(x)
    # x = Dense(units=filters[3], name='embedding')(x)
    # x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(x)
    # x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    # x = squeeze_excite_block(x)
    # x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    # x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    # x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    # model = Model(inputs=inputs, outputs=x)
    # model.summary()

### 在普通的三层卷积的自编码器中每一层卷积层后面都添加se-block，使用cifar10数据集，提取的特征数变大
    # inputs = Input(shape=[input_shape[0],input_shape[1],input_shape[2]],name='conv1_input')
    # x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    # x_1 = squeeze_excite_block(x)
    # x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x_1)
    # x_2 = squeeze_excite_block(x)
    # x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x_2)
    # # x = squeeze_excite_block(x)
    # x_3 = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv4')(x)
    # # x = squeeze_excite_block(x)
    # x = Flatten()(x_3)
    # x = Dense(units=64, name='embedding')(x)
    # x = Dense(units=filters[2]*int(input_shape[0]/16)*int(input_shape[0]/16), activation='relu')(x)
    # x = Reshape((int(input_shape[0]/16), int(input_shape[0]/16), filters[2]))(x)
    # x = Concatenate()([x, x_3])
    # # x = squeeze_excite_block(x)
    # x = Conv2DTranspose(filters[2], 3, strides=2, padding=pad3, activation='relu', name='deconv4')(x)
    # # x = squeeze_excite_block(x)
    # x = Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation='relu', name='deconv3')(x)
    # x = squeeze_excite_block(x)
    # # x = Concatenate()([x, x_2])
    # x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    # x = squeeze_excite_block(x)
    # # x = Concatenate()([x, x_1])
    # x = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    # model = Model(inputs=inputs, outputs=x)
    # model.summary()

### 在普通的三层卷积的自编码器中添加se-block和inception结构
    inputs = Input(shape=[input_shape[0], input_shape[1], input_shape[2]], name='conv1_input')
    x = Conv2D(96, 5, strides=1, padding='same', activation='relu', name='conv1')(inputs)
    x_1 = Conv2D(filters[0], 1, strides=2, padding='same', activation='relu', name='conv1_1')(x)
    x_3 = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='conv1_2')(x)
    x_5 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_3')(x)
    x = Concatenate()([x_1, x_3, x_5])
    x = Conv2D(filters[1]*2, 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    x = squeeze_excite_block(x)
    x = Flatten()(x)
    x = Dense(units=64, name='embedding')(x)
    x = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(x)
    x = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(x)
    x = squeeze_excite_block(x)
    x = Conv2DTranspose(filters[1]*2, 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    x = Conv2DTranspose(filters[0]*3, 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    x_1 = Conv2DTranspose(input_shape[2], 1, strides=2, padding='same', activation='relu', name='deconv1_1')(x)
    x_3 = Conv2DTranspose(input_shape[2], 3, strides=2, padding='same', activation='relu', name='deconv1_2')(x)
    x_5 = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', activation='relu', name='deconv1_3')(x)
    x = Concatenate()([x_1, x_3, x_5])
    x = Conv2DTranspose(input_shape[2], 5, strides=1, padding='same', name='deconv1')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()



    return model

###
# 直接使用自编码器提取到特征进行k-means聚类
###
if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('-- m', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics
    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
