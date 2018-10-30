#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import numpy as np
from core.util import read_csv_file
from transfer.image_sequence import ImageSequence
from sklearn.model_selection import train_test_split

class Transfer(object):

    def __init__(self, params):
        self._params = params

        return

    def extract_features(self, src_img, scale, patch_size, seeds):
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # print(base_model.summary())
        features = []
        for x, y in seeds:
            block= src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            feature = base_model.predict(x)
            features.append(feature)

        return features

    def load_model(self):

        return

    def fine_tuning(self, samples_name):
        csv_path = "{}/{}.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        filenames_list, labels_list = read_csv_file(self._params.PATCHS_ROOT_PATH, csv_path)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(filenames_list, labels_list, test_size=0.2, random_state=0)
        train_gen = ImageSequence(Xtrain, Ytrain, 20)
        test_gen = ImageSequence(Xtest, Ytest, 20)

        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(2, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())
        # train the model on the new data for a few epochs
        result = model.fit_generator(train_gen, steps_per_epoch=100, epochs=1, verbose=1,
                            validation_data=test_gen, validation_steps=100)
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from tensorflow.keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        result = model.fit_generator(train_gen, steps_per_epoch=100, epochs=1, verbose=1,
                            validation_data=test_gen, validation_steps=100)

