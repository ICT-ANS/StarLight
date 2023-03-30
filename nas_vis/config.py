from easydict import EasyDict
import os.path as osp
import sys

C = EasyDict()

'''path config'''
C.work_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(C.work_dir)
sys.path.append(osp.join(C.work_dir, "nas_vis", "nas_burgerformer"))

# checkpoint data log ....
C.cache_dir = osp.join(C.work_dir, "data/StarLight_Cache/nas.classification")
C.current_dir = osp.join(C.work_dir, "nas_vis")  # config.py所在的路径
C.quiver_data = osp.join(C.work_dir, "data/StarLight_Cache/quiver")
C.html_dir = osp.join(C.work_dir, "nas_vis", "nas_html")
C.qtui_dir = osp.join(C.work_dir, "qtui", "nas")
C.output_dir = osp.join(C.cache_dir, 'outputs')


if __name__ == '__main__':
    print(C.work_dir)