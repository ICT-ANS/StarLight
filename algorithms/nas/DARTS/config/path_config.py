from easydict import EasyDict
import os.path as osp
import sys

C = EasyDict()

'''path config'''
# C.work_dir = "/home/user/ANS/LightweightComputing/"
C.work_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.append(C.work_dir)

C.cache_dir = osp.join(C.work_dir, "data/LightweightComputing_Cache/nas.classification.darts") #checkpoint data log ....
C.current_dir = osp.join(C.work_dir, "nas", "classification_darts")
C.html_dir = osp.join(C.work_dir, "html", "nas", "classification_darts")
C.qtui_dir = osp.join(C.work_dir, "qtui", "classify_ui")
C.quiver_dir = osp.join(C.work_dir, "quiver")
