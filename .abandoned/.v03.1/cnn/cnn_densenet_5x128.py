import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras_preprocessing import image

from core.util import read_csv_file
from preparation.normalization import ImageNormalization
from core import *
import numpy as np
from sklearn import metrics
import os


NUM_WORKERS = 1
ROWS = 128
COLS = 128
CHANNELS = 3
nb_classes = 2
# batch_size = 32
nb_epoch = 40
input_shape = (ROWS,COLS,CHANNELS)
densenet_depth = 40
densenet_growth_rate = 12
densenet_nb_dense_block=3
nb_filter=16
class cnn_densenet_5x128(object):

    def __init__(self, params, model_name, weight_mode):
        '''
         初始化CNN分类器
        :param params: 参数
        :param model_name: 使用的模型文件
        :param samples_name: 使用的标本集的关键字（标记符），为None时是进入预测模式
        '''

        model_name = model_name
        self._params = params
        self.model_name = model_name
        self.num_classes = 2
        self.weight_mode = weight_mode

        # 无样本权重模式
        if weight_mode == 0 or weight_mode is None:
            self.class_weight = None
            self.model_root = "{}/models/{}".format(self._params.PROJECT_ROOT, model_name)
        # elif weight_mode == 1:
        #     self.class_weight = {0: 1, 1: 1, 2: 0.5, 3: 0.5}
        #     self.model_root = "{}/models/{}_W{}".format(self._params.PROJECT_ROOT, model_name, weight_mode)
        # elif weight_mode == 2:
        #     self.class_weight = {0: 0.1, 1: 0.1, 2: 1, 3: 1}
        #     self.model_root = "{}/models/{}_W{}".format(self._params.PROJECT_ROOT, model_name, weight_mode)
        # else:
        #     self.class_weight = None
        if weight_mode == 1:
            self.class_weight = {0: 1, 1: 1, 2: 0.5, 3: 0.5}
            self.model_root = "{}/models/{}_W{}".format(self._params.PROJECT_ROOT, model_name, weight_mode)
        else:
            print("Error!!!")
            return

        if (not os.path.exists(self.model_root)):
            os.makedirs(self.model_root)

        return
    def createDenseNet(self,nb_classes, input_shape, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None,
                         weight_decay=1E-4, verbose=True,model_file = None):
        ''' Build the create_dense_net model
        Args:
            nb_classes: number of classes
            img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay
        Returns: keras tensor with nb_layers of conv_block appended
        '''

        def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
            ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            Args:
                x: keras tensor
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
            Returns: keras tensor with nb_layers of conv_block appended
            '''

            def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
                ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
                Args:
                    input: Input keras tensor
                    nb_filter: number of filters
                    dropout_rate: dropout rate
                    weight_decay: weight decay factor
                Returns: keras tensor with batch_norm, relu and convolution2d added
                '''

                x = Activation('relu')(input)
                x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                                  kernel_regularizer=l2(weight_decay))(x)
                if dropout_rate is not None:
                    x = Dropout(dropout_rate)(x)

                return x

            concat_axis = 1 if K.image_dim_ordering() == "th" else -1

            feature_list = [x]

            for i in range(nb_layers):
                x = conv_block(x, growth_rate, dropout_rate, weight_decay)
                feature_list.append(x)
                x = Concatenate(axis=concat_axis)(feature_list)
                nb_filter += growth_rate

            return x, nb_filter

        def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
            ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
            Args:
                input: keras tensor
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
            Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
            '''
            concat_axis = 1 if K.image_dim_ordering() == "th" else -1

            x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                              kernel_regularizer=l2(weight_decay))(input)
            if dropout_rate is not None:
                x = Dropout(dropout_rate)(x)
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)

            x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)

            return x

        model_input = Input(shape=input_shape)

        concat_axis = 1 if K.image_dim_ordering() == "th" else -1

        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

        # layers in each dense block
        nb_layers = int((depth - 4) / 3)

        # Initial convolution
        x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
                          kernel_regularizer=l2(weight_decay))(model_input)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                beta_regularizer=l2(weight_decay))(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                       weight_decay=weight_decay)
            # add transition_block
            x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # The last dense_block does not have a transition_block
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

        densenet = Model(inputs=model_input, outputs=x)

        if verbose:
            print("DenseNet-%d-%d created." % (depth, growth_rate))
        if model_file is None:
            checkpoint_dir = self.model_root
            # latest = tf.train.latest_checkpoint(checkpoint_dir)
            latest = util.latest_checkpoint(checkpoint_dir)

            if not latest is None:
                print("loading >>> ", latest, " ...")
                densenet.load_weights(latest)
        else:
            model_path = "{}/models/{}".format(self._params.PROJECT_ROOT, model_file)
            print("loading >>> ", model_path, " ...")
            densenet.load_weights(model_path)

        return densenet

    def train_model(self, samples_name, batch_size, augmentation, epochs, initial_epoch):
            train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation)

            # checkpoint_dir = self.model_root
            checkpoint_path = "H:/yeguanglu/Pathological-images/models/densenet22_2_12_20x256/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

            cp_callback = keras.callbacks.ModelCheckpoint(
                checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
                # Save weights, every 5-epochs.
                period=1)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                              epsilon=0.0001, cooldown=0, min_lr=0)

            model = self.createDenseNet(nb_classes=nb_classes,input_shape=input_shape,depth=densenet_depth,nb_dense_block=densenet_nb_dense_block,
                  growth_rate = densenet_growth_rate,nb_filter=nb_filter)
            print(model.summary())
            # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=1e-4, rho=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            steps_per_epoch = 2
            validation_steps = 100

            model.fit_generator(train_gen, epochs=epochs, verbose=1,
                                # callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                                callbacks=[cp_callback, early_stopping, reduce_lr],
                                validation_data=test_gen, validation_steps=validation_steps, initial_epoch = initial_epoch)

            return

    def load_data(self, samples_name, batch_size, augmentation = (False, False)):
            '''
            从图片的列表文件中加载数据，到Sequence中
            :param samples_name: 列表文件的代号
            :param batch_size: 图片读取时的每批的图片数量
            :return:用于train和test的两个Sequence
            '''
            train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
            test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

            Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
            Ytrain_new = [class_id % self.num_classes for class_id in Ytrain]
            Wtrain = [self.class_weight[class_id] for class_id in Ytrain]
            train_gen = ImageSequence(Xtrain, Ytrain_new, batch_size, self.num_classes, augmentation[0])

            Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
            Ytest_new = [class_id % self.num_classes for class_id in Ytest]
            Wtest = [self.class_weight[class_id] for class_id in Ytest]
            test_gen = ImageSequence(Xtest, Ytest_new, batch_size, self.num_classes, augmentation[1])
            return  train_gen, test_gen


    def predict(self, src_img, scale, patch_size, seeds, model_file):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.createDenseNet(self,model_file)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        result = []
        for x, y in seeds:
            block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(ImageNormalization.normalize_mean(img))
            x = np.expand_dims(x, axis=0)

            predictions = model.predict(x)
            class_id = np.argmax(predictions[0])
            probability = predictions[0][class_id]
            result.append((class_id, probability))
        return result

    def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size, model_file):
            '''
            预测在种子点提取的图块
            :param src_img: 切片图像
            :param scale: 提取图块的倍镜数
            :param patch_size: 图块大小
            :param seeds: 种子点的集合
            :return:
            '''
            model = self.createDenseNet(nb_classes=nb_classes,input_shape=input_shape,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)
            optimizer = RMSprop(lr=1e-4, rho=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print(model.summary())

            image_itor = SeedSequence(src_img, scale, patch_size, seeds, batch_size)

            predictions = model.predict_generator(image_itor, verbose=1, workers=NUM_WORKERS)
            result = []
            for pred_dict in predictions:
                class_id = np.argmax(pred_dict)
                probability = pred_dict[class_id]
                result.append((class_id, probability))

            return result


    def predict_test_file(self, model_file, test_file_list, batch_size):
            model = self.createDenseNet(model_file)
            optimizer = RMSprop(lr=1e-4, rho=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print(model.summary())

            Xtest = []
            Ytest = []

            for item in test_file_list:
                test_list = "{}/{}".format(self._params.PATCHS_ROOT_PATH, item)
                Xtest1, Ytest1 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)

                Xtest.extend(Xtest1)
                Ytest.extend(Ytest1)

            image_itor = ImageSequence(Xtest, Ytest, batch_size, self.num_classes)

            predictions = model.predict_generator(image_itor, verbose=1, workers=NUM_WORKERS)
            predicted_tags = []
            predicted_probability = []
            for pred_dict in predictions:
                class_id = np.argmax(pred_dict)
                probability = pred_dict[class_id]
                predicted_tags.append(class_id)
                predicted_probability.append(probability)

            print("Classification report for classifier:\n%s\n"
                  % ( metrics.classification_report(Ytest, predicted_tags)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest, predicted_tags))

            print("average predicted probability = %s" % np.mean(predicted_probability))

            Ytest = np.array(Ytest)
            predicted_tags = np.array(predicted_tags)
            predicted_probability = np.array(predicted_probability)
            TP = np.logical_and(Ytest == 1, predicted_tags == 1)
            FP = np.logical_and(Ytest == 0, predicted_tags == 1)
            TN = np.logical_and(Ytest == 0, predicted_tags == 0)
            FN = np.logical_and(Ytest == 1, predicted_tags == 0)

            print("average TP probability = %s" % np.mean(predicted_probability[TP]))
            print("average FP probability = %s" % np.mean(predicted_probability[FP]))
            print("average TN probability = %s" % np.mean(predicted_probability[TN]))
            print("average FN probability = %s" % np.mean(predicted_probability[FN]))
            return
