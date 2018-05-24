#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-24'

"""
import  os
from feature import FeatureExtractor
from skimage import io
from sklearn.svm import NuSVC
from sklearn.externals import joblib
from sklearn import metrics

class PatchFeature(object):
    def __init__(self, params):
        self._params = params
        return

    # train_filename = "ZoneA_train.txt"
    def loading_data(self, data_filename):
        root_path = self._params.PATCHS_ROOT_PATH
        data_file = "{}/{}".format(root_path, data_filename)

        fe = FeatureExtractor.FeatureExtractor()
        features = []
        tags = []
        f = open(data_file, "r")
        for line in f:
            items = line.split(" ")
            patch_file = "{}/{}".format(root_path, items[0])
            img = io.imread(patch_file, as_grey=False)
            tag = int(items[1])
            fvector = fe.extract_glcm_feature(img)

            features.append(fvector)
            tags.append(tag)

        f.close()
        return features, tags

    # x = features, y = tags
    def train_svm(self, X, y):
        clf = NuSVC(nu=0.5, kernel='rbf', probability=True)
        rf = clf.fit(X ,y)

        model_file = self._params.PROJECT_ROOT + "/models/svm_zoneA.model"
        joblib.dump(rf, model_file)
        return clf

    def load_svm_model(self):
        model_file = self._params.PROJECT_ROOT + "/models/svm_zoneA.model"
        clf = joblib.load(model_file)

        return clf

    def test_svm(self, test_filename):
        features, expected_tags = self.loading_data(test_filename)
        classifier = self.load_svm_model()
        predicted_tags = classifier.predict(features)
        predicted_result = classifier.predict_proba(features)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected_tags, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_tags, predicted_tags))
        return predicted_result