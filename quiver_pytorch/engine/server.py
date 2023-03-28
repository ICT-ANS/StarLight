from __future__ import print_function

import json

import platform

import os, cv2
import  numpy as np
from os.path import abspath, dirname, join
import webbrowser

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS
import torch

try:
    from gevent.wsgi import WSGIServer
except ImportError:
    from gevent.pywsgi import WSGIServer


from engine.util import (
    load_img, safe_jsonify,
    validate_launch
)

from engine.model_utils import make_dot

from engine.file_utils import list_img_files, save_layer_img
from engine.vis_utils import save_layer_outputs


app = Flask(__name__)
app.threaded = True
CORS(app)

_json_graph = None
_model = None
_hook_list = None

_html_base_dir = None
_temp_folder = None
_input_folder = None

_use_gpu = False
_image_size = [224,224]

_http_server = None


def register_routes():

    global _json_graph, _model, _hook_list, _html_base_dir, _temp_folder, _input_folder, _use_gpu, _image_size
    '''
        Static Routes: only call once
    '''
    @app.route('/')
    def home():
        return send_from_directory(
            join(_html_base_dir, 'quiverboard/dist'),
            'index.html'
        )

    @app.route('/<path>')
    def get_board_files(path):
        return send_from_directory(
            join(_html_base_dir, 'quiverboard/dist'),
            path
        )

    @app.route('/temp-file/<path>')
    def get_temp_file(path):
        return send_from_directory(abspath(_temp_folder), path)

    @app.route('/input-file/<path>')
    def get_input_file(path):
        # print("--------->",abspath(_input_folder), path)
        return send_from_directory(abspath(_input_folder), path)

    '''
        Computations
    '''
    @app.route('/model')
    def get_config():
        # model_file =  "/home/user/ANS/QuiverTest/model.json"
        # with open(model_file, "r") as f:
        #     return jsonify(json.loads(f.read()))
        return jsonify(_json_graph)

    @app.route('/inputs')
    def get_inputs():
        # print (list_img_files(_input_folder))
        return jsonify(list_img_files(_input_folder))

    @app.route('/layer/<layer_name>/<input_path>')
    def get_layer_outputs(layer_name, input_path):
        print(layer_name)
        results = save_layer_outputs(_model, _hook_list, _json_graph, 
                                    layer_name, _input_folder,
                                    input_path, _temp_folder, _use_gpu, _image_size)
        return jsonify(results)
        

    @app.route('/predict/<input_path>')
    def get_prediction(input_path):
        results = [[]]
        return safe_jsonify(results)
        pass
        # print ("prediction", input_path)
        # results = [[("sa","bot_34", 0.2)],[("sa","bot_35", 0.6)]]
        # return safe_jsonify(results)


def update_model(model, hooks, input_folder, image_size, use_gpu=False, temp_folder='./tmp'):
    '''
    update model
    '''
    global _json_graph, _model, _hook_list, _html_base_dir, _temp_folder, _input_folder, _use_gpu, _image_size
    '''
        prepare model
    '''
    width = 224
    height = 224
    if image_size is not None:
        width = image_size[-1]
        height = image_size[-2]
    
    _use_gpu = use_gpu

    _image_size = [width, height]

    x = torch.zeros(1, 3, width, height, dtype=torch.float, requires_grad=False)
    
    if _use_gpu and torch.cuda.is_available():
        x = x.cuda()
        _model = model.cuda()

    _model = model

    out = _model(x)
    _json_graph = make_dot(out, params = dict(model.named_parameters()))

    _input_folder = input_folder

    _hook_list = hooks

    _temp_folder = temp_folder


def register_model(model, hooks, use_gpu, image_size, temp_folder='./tmp', input_folder='./',
            mean=None, std=None):
    '''
    :param model: the model to show
    :param classes: list of names of output classes to show in the GUI.
        if None passed - ImageNet classes will be used
    :param top: number of top predictions to show in the GUI
    :param html_base_dir: the directory for the HTML (usually inside the
        packages, quiverboard/dist must be a subdirectory)
    :param temp_folder: where the temporary image data should be saved
    :param input_folder: the image directory for the raw data
    '''
    
    update_model(model, hooks, input_folder, image_size, use_gpu=use_gpu, temp_folder=temp_folder)

    register_routes()



def launch(model, hooks, input_folder='./', use_gpu=False, image_size=None, temp_folder='./tmp', 
           port=5001, html_base_dir=None):
    if platform.system() is 'Windows':
        temp_folder = '.\\tmp'
        os.system('mkdir %s' % temp_folder)
    else:
        os.system('mkdir -p %s' % temp_folder)

    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    validate_launch(html_base_dir)

    global _html_base_dir
    _html_base_dir = html_base_dir

    register_model(model, hooks,
            use_gpu=use_gpu,
            image_size=image_size,
            temp_folder=temp_folder,
            input_folder=input_folder
        )
    
    global _http_server
    _http_server = WSGIServer(('', port), app)
    # webbrowser.open_new('http://localhost:' + str(port)) #
    _http_server.serve_forever()