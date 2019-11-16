## A Fast and Effective Detection Framework for Whole-Slide Histopathology Image Analysis
Jun Ruan, Chenchen Wu, Guanglu Ye, Jingfan Zhou, Zhikui Zhu, Junqiu Yue

>This paper presented a novel lightweight detection framework for automatic tumor detection in whole-slide histopathology images.
>

Implemented by PyTorch 1.2.0
* other requires:
    -  ghalton          0.6.1
    -  scikit-image     0.15.0
    -  scikit-learn     0.21.3
    -  Pillow           6.0.0
    -  opencv-python    4.1.0.25
    -  openslide-python 1.1.1

the dataset from Camelyon16 Grand Challenge

The core routine is in main.py; see method documentation for details

### main.py

#### setUp
    - You need to make the necessary path settings.
    - SLICES_ROOT_PATH: the directory where the Slides are located.
    - PATCHS_DICT: the directory where the extracted patches will be located.
    - NUM_WORKERS: the number of threads supported by your CPU.

#### test_patch_openslide_cancer_2k4k
    - extract the patches from tumor slides under different magnification

#### test_patch_openslide_normal
    - extract the patches from normal slides under different magnification

#### test_pack_samples_4k2k_256
    - Package the path of the training patches into a *.txt file
    - param: Samples_name is a part of the generated filename

#### test_DSC_train_model
    - train our DMC classifier

#### test_detect
    - Slide detection

#### test_update_history
    - Fine-tuning with the Slider Filter
    
#### test_calculate_ROC_pixel_level
    - Using the pathologist's annotation as ground truth, 
      ROC analysis at the pixel level were performed and the measures 
      used for comparing the algorithms were Dice coefficient, accuracy, 
      recall, F1 score and area under the ROC curve (AUC).
      
#### test_calculate_E1
    - Slide-based Evaluation in Camelyon16

#### test_calculate_E2
    - Lesion-based Evaluation in Camelyon16
