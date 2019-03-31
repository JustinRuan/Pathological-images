from time import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import keras
import numpy as np
import keras.backend as K
from keras import Input
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Lambda
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import metrics
from ConvAE import CAE


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)                                            #添加一个卷积自编码器
        hidden = self.cae.get_layer(name='embedding').output                          #从卷积自编码器中提出'embedding'层
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model，add a ClusteringLayer
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden) #添加一个聚类损失层
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])

### 使用生成对抗的方式预训练自编码器，但是并没有什么用
    def pretrainGAN(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining With GAN...')
        ############
        dense1 = Dense(32, activation='relu', name="discriminator_1")
        dense2 = Dense(32, activation='relu', name="discriminator_2")
        dense3 = Dense(1, activation='sigmoid', name="discriminator_3")

        hidden = self.cae.get_layer(name='embedding').output
        x1 = dense1(hidden)
        x1 = dense2(x1)
        discri1 = dense3(x1)

        inputs2 = Input(shape=(64,))
        x1 = dense1(inputs2)
        x1 = dense2(x1)
        discri2 = dense3(x1)

        self.cae.compile(optimizer=optimizer, loss='mse')

        lossG = Lambda(
            lambda x3: K.mean(keras.losses.categorical_crossentropy(K.ones_like(x3[0]), x3[0]), 0, keepdims=False)
                       + K.mean(keras.losses.categorical_crossentropy(K.zeros_like(x3[1]), x3[1]), 0, keepdims=False),
            name='l1_loss')([discri2, discri1])

        inputs1 = self.cae.input
        self.autoencoderG = Model(inputs=[inputs1, inputs2], outputs=lossG)
        self.autoencoderG.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

        lossA = Lambda(lambda x: K.mean(keras.losses.categorical_crossentropy(K.ones_like(x), x), 0, keepdims=False),
                       name='l2_loss')(discri1)
        self.autoencoderA = Model(inputs=inputs1, outputs=lossA)
        self.autoencoderA.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

        allModel=Model(inputs=[self.cae.input,inputs2],
                           outputs=[discri1,discri2, self.cae.output])
        plot_model(allModel, to_file=args.save_dir + '/GAN_prtrain_model.png', show_shapes=True)  # 存储dcec网络结构图

        # begin training
        t0 = time()

        for i in range(epochs):
            n_batches = int(len(x) / batch_size)
            print("------------------Epoch {}/{}------------------".format(i, epochs))
            for b in range(1, n_batches + 1):
                z_real_dist = np.random.randn(batch_size, 64) * 5.
                z_target = np.random.randn(batch_size, 1)
                batch_x = x[(b - 1) * batch_size:b * batch_size, :, :, :]
                a = self.cae.train_on_batch(batch_x, batch_x)
                # self.cae.summary()

                for s in range(28):  # 对于sedcec（cifar10） 网络
                    self.autoencoderG.layers[s].trainable = False
                self.autoencoderG.train_on_batch([batch_x, z_real_dist], z_target)
                # print("###ggggggggggggggggg#########################################")
                # self.autoencoderG.summary()
                # for x1 in self.autoencoderG.trainable_weights:
                #     print(x1.name)
                # print("############################################")
                # for x in self.autoencoderG.non_trainable_weights:
                #     print(x.name)
                for s in range(28):
                    self.autoencoderG.layers[s].trainable = True

                for s in range(27, 31):
                    self.autoencoderA.layers[s].trainable = False
                self.autoencoderA.train_on_batch(batch_x, z_target)
                # print("########AAAAAAAAAAAAAAA####################################")
                # self.autoencoderA.summary()
                # for x in self.autoencoderA.trainable_weights:
                #     print(x.name)
                # print("############################################")
                # for x in self.autoencoderA.non_trainable_weights:
                #     print(x.name)
                for s in range(27, 31):
                    self.autoencoderA.layers[s].trainable = True

                print("epoch{} batch{} loss={}".format(i, b, a))
            # a = autoencoder1.evaluate(all_test, [all_test, all_test], batch_size=100)
            # print("epoch{}  test loss={}".format(i, a))


        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

### 预训练自编码器
    def pretrain(self, x, batch_size=256, epochs=20, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(args.save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()                                          #记录当前时间
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size, save_dir=save_dir)   # 预训练卷积自编码器
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means  （使用k-means方法初始化聚类中心）
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.xx = self.encoder.predict(x)
        self.y_pred = kmeans.fit_predict(self.xx)
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        # save encode model to get feature
        print('saving encode model to:', save_dir + '/dcec_encode_model_final.h5')
        self.encoder.save(save_dir + '/dcec_encode_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'usps', 'mnist-test','cifar10'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None,choices=[None,'results/temp/pretrain_cae_model.h5','results/temp/pretrain_caei_model.h5'],help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps ,load_cifar10,load_cifar10_edge
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'cifar10':
        x, y = load_cifar10()
        # x, y = load_cifar10_edge()
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters) # 输入图片:(28, 28, 1),各个卷积层的输出的通道数:[32, 64, 128, 10]
    plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True) #存储dcec网络结构图
    dcec.model.summary()

    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,       #训练dcec
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights)
    y_pred = dcec.y_pred
    print('acc = %.4f, nmi = %.4f, ari = %.4f,ss = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred), metrics.ss(dcec.xx, y_pred)))
