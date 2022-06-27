import sys

sys.path.insert(0, '.')

from quiver.quiver_engine_v0 import server
from multiprocessing import Process
from quiver.quiver_engine_v0.model_utils import register_hook
from quiver.searched_models.model_generator import model_builder


def visualize_feature_map(model_name):
    model, input_size = model_builder(model_name)

    hook_list = register_hook(model)

    server.launch(model, hook_list, input_folder="../../quiver/data/Cat", image_size=input_size, use_gpu=False)


if __name__ == "__main__":
    p = Process(target=visualize_feature_map, args=('darts', ))
    p.start()
    while True:
        a = input("input:")
        p.terminate()
        p.join()
        print(a)
        if a == '0':
            p = Process(target=visualize_feature_map, args=('darts',))
            p.start()
        elif a == '1':
            p = Process(target=visualize_feature_map, args=('resnet18',))
            p.start()
        elif a == '2':
            p.terminate()
            p.join()
            break

