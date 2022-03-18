import numpy as np
from engine import server
from torchvision import  models
from engine.model_utils import register_hook

import threading, os

if __name__ == "__main__":
    
    model = models.vgg19(pretrained=False)
    hook_list = register_hook(model)

    rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    datapath = os.path.join(rootpath, "data/cat")

    thread = threading.Thread(target=server.launch, args=(model, hook_list, datapath, False, [200,200], ))
    thread.daemon = True
    thread.start()

    while True:
        a = input("input:")

        if a == '0':
            break
        elif a == '1':
            print("resnet")
            model = models.resnet18(pretrained=False)
            hook_list = register_hook(model) 
            datapath = os.path.join(rootpath, "data/Dog")
            server.update_model(model, hook_list, datapath, [200,200])
        elif a == '2':
            print("vgg19")
            model = models.vgg19(pretrained=False)
            hook_list = register_hook(model) 
            datapath = os.path.join(rootpath, "data/Cat")
            server.update_model(model, hook_list, datapath, [200,200])
    