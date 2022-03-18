from easydict import EasyDict
import os.path as osp
import sys

C = EasyDict()

'''path config'''
# C.work_dir = "/home/user/ANS/LightweightComputing/"
C.work_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(C.work_dir)


C.cache_dir = osp.join(C.work_dir, "data/compression")  #checkpoint data log ....
C.current_dir = osp.join(C.work_dir, "compression_vis/tmp", "ui")  # config.py所在的路径
C.html_dir = osp.join(C.work_dir, "compression_vis/tmp", "html", "modelcompress")
C.qtui_dir = osp.join(C.work_dir, "compression_vis/tmp/qtui", "compression_ui")
C.quiver_dir = osp.join(C.work_dir, "data/compression/quiver")
C.model_vis = osp.join(C.work_dir, "data/compression/model_vis")


'''hyper params'''

