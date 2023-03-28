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
