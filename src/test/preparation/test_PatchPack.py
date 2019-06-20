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
import numpy as np

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

#     #################################################################################################################
#     ####################  40 x 256  ############################
#     ##################################################################################################################
#     def test_pack_samples_4000_256(self):
#         c = Params()
#         c.load_config_file(JSON_PATH)
#
#         pack = PatchPack(c)
#
#         # data_tag = pack.initialize_sample_tags("P0430", {"S4000_256_T_cancer": 1, "S4000_256_T_normal": 0,
#         #                                                  "S4000_256_T_normal2": 0})
#         # pack.create_train_test_data(data_tag, 0, 1, "Check_P0430_4000_256", need_balance=False)
#
#         data_tag = pack.initialize_sample_tags("P0430", {"S4000_256_cancer": 1, "S4000_256_normal": 0,
#                                                                  "S4000_256_normal2": 0})
#
#         # data_tag = pack.filtering(data_tag, filter_mask=["Tumor_033", "Tumor_034",
#         #                                                  "Tumor_046","Tumor_054","Tumor_061"])
#         # data_tag = pack.initialize_sample_tags("P0430", {"S4000_256_cancer": 1,
#         #                                                  "S4000_256_normal2": 0})
#         pack.create_train_test_data(data_tag, 0.95, 0.05, "T2_P0430_4000_256", need_balance=True)
#         # 1: 0.2391 , T1_P0430_4000_256,不平衡数据集
#         # T2_P0430_4000_256, 平衡数据集
#
#
#     #################################################################################################################
#     ####################  20 x 256  ############################
#     ##################################################################################################################
#     def test_pack_samples_2000_256(self):
#         c = Params()
#         c.load_config_file(JSON_PATH)
#
#         pack = PatchPack(c)
#         data_tag = pack.initialize_sample_tags({"S2000_256_cancer":1,"S2000_256_normal":0})
#         # pack.create_data_txt(data_tag, "T_NC_2000_256")
#         pack.create_train_test_data(data_tag, 0.9, 0.1, "T_NC_2000_256")
#
# #################################################################################################################
# ####################  5 x 128  ############################
# ##################################################################################################################
#
#     def test_pack_sample_128(self):
#         c = Params()
#         c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")
#
#         pack = PatchPack(c)
#         data_tag = pack.initialize_sample_tags({"S500_128_cancer":1,"S500_128_normal":0})
#         pack.create_train_test_data(data_tag, 0.8, 0.2, "T_NC_500_128")
#         pack.create_data_txt(data_tag, "T_NC_500_128")
#
#     #################################################################################################################
#     ####################  5 x 32  ############################
#     ##################################################################################################################
#     def test_pack_sample_32(self):
#         c = Params()
#         c.load_config_file(JSON_PATH)
#
#         pack = PatchPack(c)
#         data_tag = pack.initialize_sample_tags({"S500_32_cancer":1,"S500_32_normal":0})
#         pack.create_data_txt(data_tag, "AE_500_32")
#
#     #################################################################################################################
#     ####################  10 x 256  ############################
#     ##################################################################################################################
#     def test_pack_samples_256(self):
#         c = Params()
#         c.load_config_file(JSON_PATH)
#
#         pack = PatchPack(c)
#         data_tag = pack.initialize_sample_tags({"S1000_256_cancer": (1, 0),
#                                                 "S1000_256_normal": (0, 0),
#                                                 "S2000_256_cancer": (1, 1),
#                                                 "S2000_256_normal": (0, 1),
#                                                 "S4000_256_cancer": (1, 2),
#                                                 "S4000_256_normal": (0, 2),
#                                                 })
#         pack.create_train_test_data(data_tag, 0, 1, "T_NC_x_256")



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

    def test_calc_patch_cancer_ratio(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)
        pack.calc_patch_cancer_ratio("P0619", ["S2000_256_edgeinner", "S2000_256_edgeouter", "S2000_256_cancer","S2000_256_normal",
                                               "S4000_256_edgeinner", "S4000_256_edgeouter", "S4000_256_cancer","S4000_256_normal"])
        # pack.calc_patch_cancer_ratio("P0619", ["S2000_256_edgeinner", "S2000_256_edgeouter", "S2000_256_cancer",
        #                                        "S4000_256_edgeinner", "S4000_256_edgeouter", "S4000_256_cancer"])

        # pack.calc_patch_cancer_ratio("P0619", ["S2000_256_cancer"])

    def test_read_patch_db(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        patches_code = "P0619"
        root_path = c.PATCHS_ROOT_PATH[patches_code]
        data = np.load('{}/patch_mask_ratio.npy'.format(root_path), allow_pickle=True)
        patch_db = data[()]
        print(len(patch_db.values()))
        plt.hist(patch_db.values(), bins=20, range=(0., 1),log=True)
        plt.show()

    #################################################################################################################
    ####################  40 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_4000_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)

        data_tag = pack.initialize_sample_tags_byMask("P0619", ["S4000_256_cancer", "S4000_256_normal",
                                                         "S4000_256_normal2", "S4000_256_edgeinner",
                                                         "S4000_256_edgeouter"])

        # data_tag = pack.filtering(data_tag, filter_mask=["Tumor_033", "Tumor_034",
        #                                                  "Tumor_046","Tumor_054","Tumor_061"])
        pack.create_train_test_data(data_tag, 0.95, 0.05, "T1_P0619_4000_256", need_balance=True)

    #################################################################################################################
    ####################  20 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_2000_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)

        data_tag = pack.initialize_sample_tags_byMask("P0619", ["S2000_256_cancer", "S2000_256_normal",
                                                         "S2000_256_normal2", "S2000_256_edgeinner",
                                                         "S2000_256_edgeouter"])

        # data_tag = pack.filtering(data_tag, filter_mask=["Tumor_033", "Tumor_034",
        #                                                  "Tumor_046","Tumor_054","Tumor_061"])
        pack.create_train_test_data(data_tag, 0.95, 0.05, "S1_P0619_2000_256", need_balance=True)

    #################################################################################################################
    ####################  40 + 20 x 256  ############################
    ##################################################################################################################
    def test_pack_samples_4k2k_256(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        pack = PatchPack(c)

        data_tag = pack.initialize_sample_tags_byMask("P0619", ["S4000_256_cancer", "S4000_256_normal",
                                                                "S4000_256_normal2", "S4000_256_edgeinner",
                                                                "S4000_256_edgeouter",
                                                                "S2000_256_cancer", "S2000_256_normal",
                                                                "S2000_256_normal2", "S2000_256_edgeinner",
                                                                "S2000_256_edgeouter"
                                                                ])

        # data_tag = pack.filtering(data_tag, filter_mask=["Tumor_033", "Tumor_034",
        #                                                  "Tumor_046","Tumor_054","Tumor_061"])
        pack.create_train_test_data(data_tag, 0.95, 0.05, "A1_P0619_4k2k_256", need_balance=True)