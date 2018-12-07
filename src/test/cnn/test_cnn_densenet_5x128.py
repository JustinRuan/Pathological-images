import unittest
from core import *
from cnn import cnn_densenet_5x128
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
JSON_PATH = "H:\yeguanglu\Pathological-images\config\ygl.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_simple_5x128_w(unittest.TestCase):

    def test_training(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        mode = 1

        cnn = cnn_densenet_5x128(c, "densenet40", mode)
        cnn.train_model("T_NC_500_128", batch_size = 10, augmentation = (False, False), epochs=50, initial_epoch=0)

    def test_predict(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = cnn_densenet_5x128(c, "densenet40", None)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(8768, 12192), (9152, 12128), (3328, 12032)] # C, C, S
        result = cnn.predict(imgCone, 5, 128, seeds, "simplenet128/cp-0031-0.26-0.89.ckpt")
        print(result)

        result = cnn.predict_on_batch(imgCone, 5, 128, seeds, 20, "simplenet128/cp-0031-0.26-0.89.ckpt")
        print(result)

    def test_predict_test_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = cnn_densenet_5x128(c, "densenet128", None)

        cnn.predict_test_file("densenet128/cp-0031-0.26-0.89.ckpt",
                              ["T_NC_500_128_test.txt"], 100)