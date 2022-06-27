from os.path import abspath, join

import cv2
import numpy as np
import torch

from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.file_utils import save_layer_img

# from quiver_engine.layer_result_generators import get_outputs_generator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def save_layer_outputs(model, hooks, graph, layer_name, input_folder, input_name, out_folder, input_size):

    img_cv = cv2.imread(join(abspath(input_folder), input_name))
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
    outputs = model(img_tensor)

    layers = graph["config"]["layers"]
    layer_id = None
    for layer in layers:
        if layer["name"] == layer_name:
            config = layer["config"]
            if config != "None" and "layer_id" in config:
                layer_id = config["layer_id"]
                break

    results = []
    if layer_id != None:
        for hook in hooks:
            if hook.layer_id == layer_id:
                channel = np.shape(hook.output)[1]
                max_channel = min([channel, channel])
                for channel in range(max_channel):
                    filename = save_layer_img(hook.output[0, channel, :, :], layer_name, channel, out_folder, input_name)
                    results.append(filename)
                break

    return results
