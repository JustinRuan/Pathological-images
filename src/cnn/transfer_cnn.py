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
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io, util
from sklearn.svm import NuSVC, SVC
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

class transfer_cnn(object):

    def __init__(self, params, model_name, model_filename, model_prototxt):
        self._params = params
        self.model_name = model_name
        self.model_filename = model_filename
        # self.extract_layer_name = extract_layer_name
        self.transfer_model = "{}/models/ImageNet/{}".format(self._params.PROJECT_ROOT, model_filename)
        self.deploy_proto = "{}/models/ImageNet/{}".format(self._params.PROJECT_ROOT, model_prototxt)

        # start
        if "googlenet" in model_name.lower():
            self._model_id = "googlenet"
            self.extract_layer_name = "loss3/classifier"
        elif "alexnet"in model_name.lower():
            self._model_id = "alexnet"
            self.extract_layer_name = "fc7"
        else:
            self._model_id = ""
            self.extract_layer_name = "None"
        return

    def start_caffe(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        if self._model_id == "googlenet":
            self.net = caffe.Net(self.deploy_proto, self.transfer_model, caffe.TEST)

            # 设定图片的shape格式(1,3,28,28)
            self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
            self.transformer.set_transpose('data', (2, 0, 1))
            # 减去均值，若训练时未用到均值文件，则不需要此步骤
            self.transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
            # 缩放到【0，255】之间
            self.transformer.set_raw_scale('data', 255)
            # 交换通道，将图片由RGB变为BGR
            self.transformer.set_channel_swap('data', (2, 1, 0))

            self.batch_count = self.net.blobs['data'].num
        else:
            pass
        return

    def loading_data(self, data_list):
        root_path = self._params.PATCHS_ROOT_PATH
        data_file = "{}/{}".format(root_path, data_list)

        self.start_caffe()

        fe = FeatureExtractor.FeatureExtractor()
        glcm_features = []
        cnn_features = []
        tags = []
        n = 0

        f = open(data_file, "r")
        sizehint = len(f.readline()) * self.batch_count - 1
        f.seek(0)

        while 1:
            lines = f.readlines(sizehint)
            if not lines:
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

                # img2 = caffe.io.load_image(patch_file)  # 加载图片
                images.append(self.transformer.preprocess('data', util.img_as_float(img).astype(np.float32)))

            # 用全0矩阵 补齐不足一个批次的不足部分
            img_count = len(lines)
            if img_count < self.batch_count:
                aligned_count = self.batch_count - img_count
                for i in range(aligned_count):
                    images.append(self.transformer.preprocess('data', np.zeros(img.shape)))

            self.net.blobs['data'].data[...] = images
            output = self.net.forward()

            for index, fvector2 in enumerate(self.net.blobs[self.extract_layer_name].data):
                if index >= img_count:
                    break
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
        X = self.comobine_features(glcm_features, cnn_features, top_index)

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

        glcm_features, cnn_features, expected_tags = self.loading_data(test_filename)
        features = self.comobine_features(glcm_features, cnn_features)

        classifier = self.load_svm_model()
        predicted_tags = classifier.predict(features)

        # predicted_result = classifier.predict_proba(features)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected_tags, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_tags, predicted_tags))
        # return predicted_result

    def comobine_features(self, glcm_features, cnn_features, top_index = []):
        if len(top_index) == 0:
            index_filename = "{}_{}.npy".format(self.transfer_model, "top_index")
            top_index = np.load(index_filename)

        new_cnn_features = np.array(cnn_features)[:, top_index]
        features = np.column_stack((glcm_features, new_cnn_features))
        return features

    # img_list 是 io.imread得到图像的List
    def extract_cnn_feature(self, src_img_list):
        fe = FeatureExtractor.FeatureExtractor()
        glcm_features = []
        cnn_features = []

        img_count = len(src_img_list)

        if img_count > self.batch_count  :
            prior = 0
            for next in range(self.batch_c, img_count, self.batch_count):
                glcm_subset, cnn_subset = self.extract_cnn_feature(src_img_list[prior:next])
                glcm_features.extend(glcm_subset)
                cnn_features.extend(cnn_subset)
                prior = next

            if prior < img_count:
                glcm_subset, cnn_subset = self.extract_cnn_feature(src_img_list[prior:])
                glcm_features.extend(glcm_subset)
                cnn_features.extend(cnn_subset)

            return glcm_features, cnn_features
        else:
            images = []
            for img in src_img_list:
                fvector1 = fe.extract_glcm_feature(img)
                glcm_features.append(fvector1)

                images.append(self.transformer.preprocess('data', util.img_as_float(img).astype(np.float32)))

            # 用全0矩阵 补齐不足一个批次的不足部分
            if img_count < self.batch_count:
                aligned_count = self.batch_count - img_count
                for i in range(aligned_count):
                    images.append(self.transformer.preprocess('data', np.zeros(img.shape)))

            self.net.blobs['data'].data[...] = images
            output = self.net.forward()

            for index, fvector2 in enumerate(self.net.blobs[self.extract_layer_name].data):
                if index >= img_count:
                    break
                cnn_features.append(fvector2.copy()) # 神坑，花费一天时间爬出

            return glcm_features, cnn_features
