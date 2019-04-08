#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import os
import unittest
from core import Params
from preparation import PatchPack
import matplotlib.pyplot as plt
from skimage.io import imread
from core.util import read_csv_file

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestPatchPack(unittest.TestCase):

    # def test_pack_refine_sample_tags_SC(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #
    #     pack = PatchPack(c)
    #     pack.refine_sample_tags_SVM({"S500_128_cancer":1,"S500_128_stroma":0}, "R_SC_5x128")
    #
    #
    # def test_extract_refine_sample_SC(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #
    #     pack = PatchPack(c)
    #     dir_map = {"S500_128_cancer":1,"S500_128_stroma":0}
    #     pack.extract_refine_sample_SC(5, dir_map, "R_SC_5x128", 128)
    #
    #     # dir_map = {"S500_128_edge": 1, "S500_128_lymph": 0}
    #     # pack.extract_refine_sample_LE(5, dir_map,  "R_SC_5x128", 128)
    #
    # def test_packing_refined_samples(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #
    #     pack = PatchPack(c)
    #     pack.packing_refined_samples(5, 128)

    #################################################################################################################
    ####################  40 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_4000_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        data_tag = pack.initialize_sample_tags("Target", {"S4000_256_cancer": 1, "S4000_256_normal": 0})
        # pack.create_data_txt(data_tag, "T_NC_P0404_4000_256")
        # pack.create_train_test_data(data_tag, 0.9, 0.1, "T_NC_Simple0327_2_4000_256")
        pack.create_train_test_data(data_tag, 0, 1, "Target_T1_4000_256")

    #################################################################################################################
    ####################  20 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_2000_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        data_tag = pack.initialize_sample_tags({"S2000_256_cancer":1,"S2000_256_normal":0})
        # pack.create_data_txt(data_tag, "T_NC_2000_256")
        pack.create_train_test_data(data_tag, 0.9, 0.1, "T_NC_2000_256")

#################################################################################################################
####################  5 x 128  ############################
##################################################################################################################

    def test_pack_sample_128(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")

        pack = PatchPack(c)
        data_tag = pack.initialize_sample_tags({"S500_128_cancer":1,"S500_128_normal":0})
        pack.create_train_test_data(data_tag, 0.8, 0.2, "T_NC_500_128")
        pack.create_data_txt(data_tag, "T_NC_500_128")

    #################################################################################################################
    ####################  5 x 32  ############################
    ##################################################################################################################
    def test_pack_sample_32(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        data_tag = pack.initialize_sample_tags({"S500_32_cancer":1,"S500_32_normal":0})
        pack.create_data_txt(data_tag, "AE_500_32")

    #################################################################################################################
    ####################  10 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        data_tag = pack.initialize_sample_tags({"S1000_256_cancer": (1, 0),
                                                "S1000_256_normal": (0, 0),
                                                "S2000_256_cancer": (1, 1),
                                                "S2000_256_normal": (0, 1),
                                                "S4000_256_cancer": (1, 2),
                                                "S4000_256_normal": (0, 2),
                                                })
        pack.create_train_test_data(data_tag, 0, 1, "T_NC_x_256")


    def test_extract_features_save_file(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")

        pack = PatchPack(c)
        pack.extract_feature_save_file("T_NC_500_128")
        pack.train_SVM("T_NC_500_128")

    def test_train_SVM_for_refine_sample(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")

        pack = PatchPack(c)
        pack.train_SVM_for_refine_sample("T_NC_500_128")

    def test_create_refined_sample_txt(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")

        pack = PatchPack(c)
        dir_map = {"S500_128_cancer": 1, "S500_128_normal": 0}
        tag_name_map = {1 : "cancer", 0 : "normal"}
        pack.create_refined_sample_txt(5, 128, dir_map, tag_name_map, "T_NC_500_128")


    def test_show_sample_txt(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")
        sample_txt = "{}/{}".format(c.PATCHS_ROOT_PATH, "S500_128_False_normal.txt")
        patch_path = c.PATCHS_ROOT_PATH

        filenames_list, labels_list = read_csv_file(patch_path, sample_txt)

        fig = plt.figure(figsize=(8,10), dpi=100)
        for index, filename in enumerate(filenames_list):
            img = imread(filename)
            pos = index % 20
            plt.subplot(4, 5, pos + 1)
            plt.imshow(img)
            plt.axis("off")

            if pos == 19:
                fig.tight_layout()  # 调整整体空白
                plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
                plt.show()

                # os.system('pause')
            # if pos == 19:
            #     break;

    def test_packing_refined_samples(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")

        pack = PatchPack(c)
        # dir_map = {"S500_128_True_cancer.txt": 1, "S500_128_True_normal.txt": 0,
        #            "S500_128_False_cancer.txt": 3, "S500_128_False_normal.txt": 2}
        # pack.packing_refined_samples(dir_map, 5, 128)

        dir_map = {"S500_128_True_cancer.txt": 1, "S500_128_True_normal.txt": 0}
        pack.packing_refined_samples(dir_map, 5, 128, "TrueS")

    ###############################################################################################################
    # Multiple scale combination (MSC)
    ###############################################################################################################
    def test_pack_samples_MSC(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        pack.create_train_test_data_MSC({10: "S1000", 20:"S2000", 40:"S4000"},
                                               {"256_cancer":       (1, 3),
                                                "256_edgeinner":    (1, 2),
                                                "256_edgeouter":    (0, 1),
                                                "256_normal":       (0, 0),
                                                },
                                                   0, 1, "T_NC2_msc_256")

