#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-10-20'

"""

import datetime
import os
import random

import numpy as np
from pytorch.cancer_map import CancerMapBuilder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, naive_bayes, ensemble
from sklearn.model_selection import GridSearchCV
import joblib
from skimage import morphology
from skimage.morphology import square
from core.lagrangian_s3vm import lagrangian_s3vm_train

class SlidePredictor(object):
    def __init__(self, params, ):
        self._params = params

    def extract_slide_features(self, tag =0, normal_names = None, tumor_names = None):
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        if tag == 0:
            code = "_history.npz"
        else:
            code = "_history_v{}.npz".format(tag)

        K = len(code)

        feature_data = []
        label_data = []

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if (slice_id not in normal_names) and (slice_id not in tumor_names):
                continue

            if ext_name == ".npz" and code in result_file:
                print("loading data : {}, {}".format(slice_id, result_file))
                result = np.load("{}/{}".format(save_path, result_file), allow_pickle=True)
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()
                feature = self.calculate_slide_features(history, x1, y1, x2, y2)

                print(slice_id, ":\t", feature)

                if slice_id in normal_names:
                    feature_data.append(feature)
                    label_data.append(0)
                elif slice_id in tumor_names:
                    feature_data.append(feature)
                    label_data.append(1)

        return feature_data, label_data

    def save_train_data(self, feature_data, label_data, filename, append = False):
        print("len =", len(feature_data), len(label_data))
        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, filename)
        if not append:
            np.savez_compressed(filename, x=feature_data, y=label_data,)
        else:
            result = np.load(filename, allow_pickle=True)
            x = list(result["x"])
            y = list(result["y"])
            x.extend(feature_data)
            y.extend(label_data)
            np.savez_compressed(filename, x=x, y=y, )

    def calculate_slide_features(self, history, x1, y1, x2, y2):
        mode = 2
        if mode == 1: # by cancer map
            cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
            cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

            selem_size = 8
            cancer_map = morphology.erosion(cancer_map,square(selem_size))

            data = cancer_map.ravel()
        elif mode == 2: # by history
            value = np.array(list(history.values()))
            data = 1 / (1 + np.exp(-value))

        data = data[data >=0.5]

        DIM = 5
        L = len(data)
        if L > 0:
            max_prob = np.max(data)
            hist = np.histogram(data, bins=DIM - 2, range=(0.5,1), density=False)
            feature = np.array(hist[0], dtype=np.float) / L
            feature = np.append(feature, [L, max_prob])
            return feature
        else:
            return np.zeros((DIM,))

    # def train_svm(self,file_name, test_name):
    #     X_test, X_train, y_test, y_train = self.read_data(file_name, test_name, mode=1)
    #
    #     max_iter = 5000
    #     model_params = [{'C': 0.0001}, {'C': 0.001}, {'C': 0.01}, {'C': 0.1},
    #                     {'C': 0.5}, {'C': 1.0}, {'C': 1.2}, {'C': 1.5},
    #                     {'C': 2.0}, {'C': 10.0},]
    #
    #     result = {'pred': None, 'score': 0, 'clf': None}
    #
    #     feature_num = len(X_train[0])
    #     for params in model_params:
    #         # 'linear', 'poly', 'rbf', 'sigmoid'
    #         clf = svm.SVC(**params,kernel='linear', class_weight={0:1, 1:1}, probability=True, gamma=1e-1,
    #                       max_iter=max_iter, verbose=0, ) # class_weight='balanced', class_weight={0:1, 1:1}
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_train)
    #         score = metrics.accuracy_score(y_train, y_pred)
    #         print('feature num = {}, C={:8f} => score={:5f}'.format(feature_num, params['C'], score))
    #
    #         if score > result["score"]:
    #             result = {'pred': y_pred, 'score': score, 'clf': clf}
    #
    #     print("the best score = {}".format(result["score"]))
    #
    #     target_names = ['normal', 'cancer']
    #     report = metrics.classification_report(y_train, result["pred"], target_names=target_names, digits = 4, output_dict=True)
    #     print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, result["pred"]))
    #
    #     self.save_model(result["clf"], "linearsvm", X_test, y_test)
    #
    #     return

    def train_svm(self,file_name, test_name):
        X_test, X_train, y_test, y_train = self.read_data(file_name, test_name, mode=1)

        max_iter = 10000

        #'kernel': ('linear'),
        parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.2, 1.5, 2.0, 10]}
        # parameters = { 'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.2, 1.5, 2.0, 10],
        #               'gamma':[2, 1, 0.1, 0.5, 0.01, 0.001, 0.0001]}

        svc = svm.SVC(kernel='linear', probability=True,max_iter=max_iter,verbose=0,)
        grid  = GridSearchCV(svc, parameters, cv=5)
        grid .fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        clf = grid.best_estimator_
        self.save_model(clf, "linearsvm", X_test, y_test)

        return

    def save_model(self, clf, typename, X_test, y_test):
        print(clf)
        y_pred = clf.predict_proba(X_test)
        prob = y_pred[:,1]

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, prob)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

        y_test[y_test == -1] = 0 # for S3VM
        pred = prob > 0.5
        accu = metrics.accuracy_score(y_test,pred)
        recall = metrics.recall_score(y_test,pred)
        f1 = metrics.f1_score(y_test,pred)
        print("roc auc  = {:.4f}, accu = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(roc_auc,accu, recall, f1), )

        if roc_auc > 0.9:
            model_file = self._params.PROJECT_ROOT + "/models/slide_predictor_{}_{:.4f}_{:.4f}.model".format(typename,
                                                                                                             roc_auc,
                                                                                                             accu)
            joblib.dump(clf, model_file)
            print("saved : ", model_file)
            for fp, tp in zip(false_positive_rate, true_positive_rate):
                print(fp, '\t', tp)


    def read_data(self, file_name, test_filename, mode):
        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, file_name)
        result = np.load(filename, allow_pickle=True)
        x = result["x"]
        y = result["y"]
        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, test_filename)
        result = np.load(filename, allow_pickle=True)
        if mode == 1:
            # rand = random.randint(1, 100)
            rand = 100
            print("rand", rand)
            x = np.append(x, result["x"], axis=0)
            y = np.append(y, result["y"], axis=0)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                              random_state=rand)
        elif mode==2:
            tx = result["x"]
            ty = result["y"]
            X_train, X_test, y_train, y_test = x, tx, y, ty
        else:
            tx = result["x"]
            ty = result["y"]
            x = np.append(x, result["x"], axis=0)
            y = np.append(y, result["y"], axis=0)
            X_train, X_test, y_train, y_test = x, tx, y, ty

        return X_test, X_train, y_test, y_train

    def train_test(self, file_name, test_name):
        X_test, X_train, y_test, y_train = self.read_data(file_name,test_name, mode = 1)

        # clf = naive_bayes.GaussianNB(var_smoothing=1e-3)
        clf = naive_bayes.MultinomialNB(alpha=0.01)
        clf.fit(X_train, y_train)

        self.save_model(clf, "MNB", X_test, y_test)

    def train_s3vm(self,file_name, test_name):
        X_test, X_train, y_test, y_train = self.read_data(file_name, test_name, mode=1)
        pos_count = np.sum(y_train)
        y_test[y_test == 0] = -1
        y_train[y_train == 0] = -1
        rdm = np.random.RandomState()
        l = len(y_train)  # labeled samples
        u = len(y_test)  # unlabeled ones
        r = float(pos_count) / len(y_train)  # positive samples ratio
        max_iter = 5000

        xtrain_l, ytrain_l, xtrain_u = X_train, y_train, X_test
        svc = svm.SVC(kernel='rbf', probability=True, gamma=1e-2, C=1.0,
                          max_iter=max_iter, verbose=0, )
        svc.fit(X_train, y_train)
        # train the semi-supervised model
        lagr_s3vc = lagrangian_s3vm_train(xtrain_l,
                                          ytrain_l,
                                          xtrain_u,
                                          svc,
                                          r=r,
                                          batch_size=2000,
                                          rdm=rdm)

        self.save_model(lagr_s3vc, "S3VM", X_test, y_test)

    def data_augment(self, file_name, output_filename, count):
        def merge(fa, fb):
            # feature: histogram[1-3], area, max_prob
            hist_a, area_a, p_a = fa[0:3], fa[3], fa[4]
            hist_b, area_b, p_b = fb[0:3], fb[3], fb[4]

            total_area = area_a + area_b
            max_prob = max(p_a, p_b)

            hist_count_a = hist_a * area_a
            hist_count_b = hist_b * area_b
            hist = (hist_count_a + hist_count_b) / total_area
            feature = np.append(hist, [total_area, max_prob])
            return feature

        # X_test, X_train, y_test, y_train = self.read_data(file_name, mode=2)
        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, file_name)
        result = np.load(filename, allow_pickle=True)
        X_train = result["x"]
        y_train = result["y"]
        train_len = len(y_train)

        aug_features = []
        aug_label = []

        for i in range(count):
            a = random.randint(0, train_len - 1)
            b = random.randint(0, train_len - 1)

            label_a, label_b = y_train[a], y_train[b]
            feature_a, feature_b = X_train[a], X_train[b]
            if a!=b and not (np.all(feature_a==0) or np.all(feature_b==0)):
                aug_feat = merge(feature_a, feature_b)

                if label_a == 0 and label_b == 0:
                    aug_features.append(aug_feat)
                    aug_label.append(0)
                else:
                    aug_features.append(aug_feat)
                    aug_label.append(1)

        X_train = np.append(X_train, aug_features, axis=0)
        y_train = np.append(y_train, aug_label, axis=0)
        x = []
        y = []
        for i in range(len(y_train)):
            feature = X_train[i]
            label = y_train[i]
            if np.all(feature==0) and label == 1:
                continue
            x.append(feature)
            y.append(label)

        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, output_filename)
        np.savez_compressed(filename, x=x, y=y, )
        print("len x = ", len(x), "len y = ", len(y))
        return