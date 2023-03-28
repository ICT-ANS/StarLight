import re
from os.path import relpath, abspath
from os import listdir
import imageio
import cv2
import numpy as np
# from scipy.misc import imsave

from engine.util import deprocess_image

def save_layer_img(layer_outputs, layer_name, idx, temp_folder, input_path):
    filename = get_output_filename(layer_name, idx, temp_folder, input_path)

    src = (deprocess_image(layer_outputs)*255).astype(np.uint8)

    # applyColorMap 可选参数
    # COLORMAP_AUTUMN = 0,
    # COLORMAP_BONE = 1,
    # COLORMAP_JET = 2,
    # COLORMAP_WINTER = 3,
    # COLORMAP_RAINBOW = 4,
    # COLORMAP_OCEAN = 5,
    # COLORMAP_SUMMER = 6,
    # COLORMAP_SPRING = 7,
    # COLORMAP_COOL = 8,
    # COLORMAP_HSV = 9,
    # COLORMAP_PINK = 10,
    # COLORMAP_HOT = 11
    src = cv2.applyColorMap(src, cv2.COLORMAP_JET)

    imageio.imwrite(filename, src)

    return relpath(filename, abspath(temp_folder))
    
def get_output_filename(layer_name, z_idx, temp_folder, input_path):
    return '{}/{}_{}_{}.png'.format(temp_folder, layer_name, str(z_idx), input_path)

def list_img_files(input_folder):
    image_regex = re.compile(r'.*\.(jpg|png|gif)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]
