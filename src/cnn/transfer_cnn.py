#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-30'

"""
from core import Params
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from skimage import io
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io
from sklearn.svm import NuSVC, SVC
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

class transfer_cnn(object):

    def __init__(self, params, model_filename, model_prototxt,extract_layer_name):
        self._params = params
        self.model_filename = model_filename
        self.extract_layer_name = extract_layer_name
        self.transfer_model = "{}/models/ImageNet/{}".format(self._params.PROJECT_ROOT, model_filename)
        self.deploy_proto = "{}/models/ImageNet/{}".format(self._params.PROJECT_ROOT, model_prototxt)

        return

    def loading_data(self, data_list):
        root_path = self._params.PATCHS_ROOT_PATH
        data_file = "{}/{}".format(root_path, data_list)

        caffe.set_device(0)
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy_proto, self.transfer_model, caffe.TEST)

        # 设定图片的shape格式(1,3,28,28)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
        transformer.set_transpose('data', (2, 0, 1))
        # 减去均值，若训练时未用到均值文件，则不需要此步骤
        # transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
        # 缩放到【0，255】之间
        transformer.set_raw_scale('data', 255)
        # 交换通道，将图片由RGB变为BGR
        transformer.set_channel_swap('data', (2, 1, 0))

        batch_count = net.blobs['data'].num

        fe = FeatureExtractor.FeatureExtractor()
        glcm_features = []
        cnn_features = []
        tags = []
        n = 0

        f = open(data_file, "r")

        sizehint = len(f.readline()) * batch_count - 1
        f.seek(0)

        while 1:
            lines = f.readlines(sizehint)
            if not lines or len(lines)!=batch_count:
                break
            images = []
            for line in lines:
                items = line.split(" ")
                patch_file = "{}/{}".format(self._params.PATCHS_ROOT_PATH, items[0])
                tag = int(items[1])
                tags.append(tag)

                img = io.imread(patch_file, as_grey=False)
                fvector1 = fe.extract_glcm_feature(img)
                glcm_features.append(fvector1)

                img2 = caffe.io.load_image(patch_file)  # 加载图片
                images.append(transformer.preprocess('data', img2))

            net.blobs['data'].data[...] = images
            output = net.forward()

            for fvector2 in net.blobs[self.extract_layer_name].data:
                cnn_features.append(fvector2.copy()) # 神坑，花费一天时间爬出

                print("{} --> tag:{}, f1:{}, f2:{}".format(n, tags[n], glcm_features[n][0:3],fvector2[0:4]))
                n += 1

        f.close()
        # np.save("features",{"cnn":cnn_features,"tags":tags})
        # np.savetxt('features.txt', np.column_stack((tags,cnn_features)), delimiter=',',fmt='%.6e')
        return glcm_features, cnn_features, tags

    def select_features(self, cnn_features, tags):
        data = np.column_stack((tags, cnn_features))
        df = pd.DataFrame(data)

        diff = df.groupby(0).mean()
        # print(diff)
        absDiff = abs(diff.iloc[0,:] - diff.iloc[1,:]).sort_values(ascending=False)
        # print(abs(diff.iloc[0,:] - diff.iloc[1,:]))
        print(absDiff[:100])

        top_index = absDiff[:100].index.values - 1 # 将index 从1开始的，0是tag的位置
        index_filename = "{}_{}".format(self.transfer_model, "top_index")
        np.save(index_filename, top_index)
        return top_index

    # x = features, y = tags
    def train_svm(self, glcm_features, cnn_features, tags, top_index):
        y = tags

        new_cnn_features = np.array(cnn_features)[:, top_index]
        X = np.column_stack((glcm_features, new_cnn_features))
        # X = new_cnn_features

        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[0,10]}
        clf = GridSearchCV(SVC(), parameters, cv=5)

        rf = clf.fit(X ,y)
        print("Best param: {} \t Beat score: {}".format(clf.best_params_, clf.best_score_))

        model_file = "{}/models/svm_{}.model".format(self._params.PROJECT_ROOT, self.model_filename)
        joblib.dump(rf, model_file)
        return clf

    def load_svm_model(self):
        model_file = "{}/models/svm_{}.model".format(self._params.PROJECT_ROOT, self.model_filename)
        clf = joblib.load(model_file)

        return clf

    def test_svm(self, test_filename):

        index_filename = "{}_{}.npy".format(self.transfer_model, "top_index")
        top_index = np.load(index_filename)
        classifier = self.load_svm_model()

        glcm_features, cnn_features, expected_tags = self.loading_data(test_filename)

        new_cnn_features = np.array(cnn_features)[:, top_index]
        features = np.column_stack((glcm_features, new_cnn_features))
        # features = new_cnn_features
        predicted_tags = classifier.predict(features)

        # predicted_result = classifier.predict_proba(features)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected_tags, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_tags, predicted_tags))
        # return predicted_result