import threading,os
import numpy as np
from torchvision import models
from quiver.quiver_engine_v0 import server
from quiver.quiver_engine_v0.model_utils import register_hook
from quiver.searched_models.model_generator import model_builder


def visualize_feature_map_in_one_thread(model_name, quiver_dir, first_init):
    model, input_size = model_builder(model_name)
    hook_list = register_hook(model)

    input_folder = os.path.join(quiver_dir, "data/mars")

    if first_init:
        thread = threading.Thread(target=server.launch, args=(model, hook_list, input_folder, False, [200, 200],))
        thread.daemon = True
        thread.start()
    else:
        server.update_model(model, hook_list, input_folder, [200, 200])


if __name__ == "__main__":

    model = models.vgg19(pretrained=False)
    hook_list = register_hook(model)

    thread = threading.Thread(target=server.launch, args=(model, hook_list, "./data/Cat", False, [200, 200],))
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
            server.update_model(model, hook_list, "./data/Dog", [200, 200])
        elif a == '2':
            print("vgg19")
            model = models.vgg19(pretrained=False)
            hook_list = register_hook(model)
            server.update_model(model, hook_list, "./data/Cat", [200, 200])
