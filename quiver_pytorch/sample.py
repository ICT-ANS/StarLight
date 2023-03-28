'''
Author: mjlsuccess 1018584208@qq.com
Date: 2023-03-21 19:07:52
LastEditors: mjlsuccess 1018584208@qq.com
LastEditTime: 2023-03-21 19:10:30
FilePath: \quiver\sample.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from engine import server
from engine.model_utils import register_hook
from torchvision import models

import os

if __name__ == "__main__":
    model = models.alexnet(pretrained=False)

    hook_list = register_hook(model)

    rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    datapath = os.path.join(rootpath, "data/cat")

    server.launch(model, hook_list, input_folder=datapath, image_size=[250, 250],
                  use_gpu=False, port=5001)
