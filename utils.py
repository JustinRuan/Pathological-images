#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

# Cancer = 1
# Normal = 0

"""
   1. 最高分辨率图像使用scale = 20进行采集，这时每个Patch边长为256
   2. 全图使用scale = 2.5进行采集，对应的Patch的边长为32
"""
# googleNet
EXTRACT_SCALE = 20
PATCH_SIZE_HIGH = 256
PATCH_SIZE_LOW = 16

# lenet
# EXTRACT_SCALE = 8
# PATCH_SIZE_HIGH = 32
# PATCH_SIZE_LOW = 4

GLOBAL_SCALE = PATCH_SIZE_LOW / PATCH_SIZE_HIGH * EXTRACT_SCALE  # when lenet , = 1, when googleNet， = 1.25
AMPLIFICATION_SCALE = PATCH_SIZE_HIGH / PATCH_SIZE_LOW  # when lenet , = 8, when googleNet， = 16

EXTRACT_PATCH_DIST = 4
CLASSIFY_PATCH_DIST = 8

# 切出来的图块的存储路径
PATCH_PATH_CANCER = "D:/Study/breast/Patches/cancer"
PATCH_PATH_NORMAL = "D:/Study/breast/Patches/normal"

# 数字切片DLL所在路径
KFB_SDK_PATH = "D:/CloudSpace/DoingNow/WorkSpace/lib/KFB_SDK"

# 数字切片所在路径
SLIDES_PATH = "D:/Study/breast/3Plus"

# 项目所在路径，网络模型也在这个路径的/Pathological_Images/DetectCancer/models下
PROJECT_PATH = "D:/CloudSpace/DoingNow/WorkSpace"

# 训练测试网络用的切片所在路径
TRAIN_TEST_PATCHES_PATH = "D:/Study/breast/Patches/S20_P256"
