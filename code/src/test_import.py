import sys
import os
sys.path.append('/home/dyx2/team2box/team2box')
sys.path.append('/home/dyx2/team2box/team2box/aaaidata/knrm/model')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只用 GPU:2

import tensorflow as tf
print("当前 TensorFlow 版本：", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("成功设置 GPU 内存按需分配")
    except RuntimeError as e:
        print("GPU 设置失败：", e)

try:
    from team2box.aaaidata.knrm.model import model_knrm
    print("模块导入成功！")
    print("实际加载的 model_knrm 文件：", model_knrm.__file__)
except Exception as e:
    print("模块导入失败：", e)

