from __future__ import absolute_import, division, print_function
import json
from flask.json import jsonify
import numpy as np
# from engine.imagenet_utils import preprocess_input, decode_imagenet_predictions
from os import path
import cv2



def validate_launch(html_base_dir):
    print('Starting webserver from:', html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard')), 'Quiverboard must be a ' \
                                                                       'subdirectory of {}'.format(html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard', 'dist')), 'Dist must be a ' \
                                                                               'subdirectory of quiverboard'
    assert path.exists(
        path.join(html_base_dir, 'quiverboard', 'dist', 'index.html')), 'Index.html missing'


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x

def load_img_scaled(input_path, target_shape, grayscale=False):
    img = cv2.imread(input_path)
    img = cv2.resize(img, target_shape) / 255.0

    

    return np.expand_dims(
        image.img_to_array(image.load_img(input_path, target_size=target_shape, grayscale=grayscale)) / 255.0,
        axis=0
    )

def load_img(input_path, target_shape, grayscale=False, mean=None, std=None):

    img = image.load_img(input_path, target_size=target_shape,
                         grayscale=grayscale)
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)
    if not grayscale:
        img_arr = preprocess_input(img_arr, mean=mean, std=std)
    return img_arr


def get_jsonable_obj(obj):
    return json.loads(get_json(obj))

def get_json(obj):
    return json.dumps(obj, default=get_json_type)

def safe_jsonify(obj):
    return jsonify(get_jsonable_obj(obj))

def get_json_type(obj):

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        return obj.item()

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable')
