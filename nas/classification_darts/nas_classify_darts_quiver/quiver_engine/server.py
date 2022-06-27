from __future__ import print_function

import json

import platform

import os, cv2
import numpy as np
import time
from datetime import datetime
from gevent import socket
from os.path import abspath, dirname, join
import webbrowser

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS
import torch

try:
    from gevent.wsgi import WSGIServer
except ImportError:
    from gevent.pywsgi import WSGIServer, WSGIHandler
from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.util import (load_img, safe_jsonify, validate_launch)

from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.model_utils import make_dot

from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.file_utils import list_img_files, save_layer_img
from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.vis_utils import save_layer_outputs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_app(model, hooks, classes, top, input_size, html_base_dir, temp_folder='./tmp', input_folder='./', mean=None, std=None):
    '''
    The base of the Flask application to be run
    :param model: the model to show
    :param classes: list of names of output classes to show in the GUI.
        if None passed - ImageNet classes will be used
    :param top: number of top predictions to show in the GUI
    :param html_base_dir: the directory for the HTML (usually inside the
        packages, quiverboard/dist must be a subdirectory)
    :param temp_folder: where the temporary image data should be saved
    :param input_folder: the image directory for the raw data
    :param mean: list of float mean values
    :param std: lost of float std values
    :return:
    '''

    # single_input_shape, input_channels = get_input_config(model)
    app = Flask(__name__)
    app.threaded = True
    CORS(app)
    '''
        prepare model
    '''
    x = torch.zeros(input_size, dtype=torch.float, requires_grad=False).to(device)
    model.to(device)
    model.eval()
    out = model(x)
    print(1)

    graph = make_dot(out, params=dict(model.named_parameters()))
    '''
        Static Routes
    '''
    @app.route('/')
    def home():
        return send_from_directory(join(html_base_dir, 'quiverboard/dist'), 'index.html')

    @app.route('/<path>')
    def get_board_files(path):
        return send_from_directory(join(html_base_dir, 'quiverboard/dist'), path)

    @app.route('/temp-file/<path>')
    def get_temp_file(path):
        return send_from_directory(abspath(temp_folder), path)

    @app.route('/input-file/<path>')
    def get_input_file(path):
        return send_from_directory(abspath(input_folder), path)

    '''
        Computations
    '''

    @app.route('/model')
    def get_config():
        # print (jsonify(json.loads(model.to_json())))
        # print("test-------------")

        # model_file =  "/home/user/ANS/QuiverTest/model.json"
        # model_file = "/home/user/ANS/pytorch_model_vis/model_1.json"
        # with open(model_file, "r") as f:
        #     return jsonify(json.loads(f.read()))
        return jsonify(graph)

    @app.route('/inputs')
    def get_inputs():
        return jsonify(list_img_files(input_folder))

    @app.route('/layer/<layer_name>/<input_path>')
    def get_layer_outputs(layer_name, input_path):
        print(layer_name, input_path)

        results = save_layer_outputs(model, hooks, graph, layer_name, input_folder, input_path, temp_folder, input_size=tuple(input_size[2:]))
        print("------------------ssss---------------------")
        return jsonify(results)

    @app.route('/predict/<input_path>')
    def get_prediction(input_path):
        # print ("prediction", input_path)
        #results = [[("sa", "bot_34", 0.2)], [("sa", "bot_35", 0.6)]]
        results = []
        return safe_jsonify(results)

    return app


def run_app(app, port=5000):
    http_server = WSGIServer(('', port), app)
    # webbrowser.open_new('http://localhost:' + str(port)) #
    http_server.serve_forever()


def launch(model, hooks, input_folder='./', input_size=[1, 3, 32, 32], classes=None, top=5, temp_folder='./tmp', port=5000, html_base_dir=None, mean=None, std=None):
    if platform.system() is 'Windows':
        temp_folder = '.\\tmp'
        os.system('mkdir %s' % temp_folder)
    else:
        os.system('mkdir -p %s' % temp_folder)

    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    # print(html_base_dir)
    validate_launch(html_base_dir)

    return run_app(get_app(model, hooks, classes, top, input_size=input_size, html_base_dir=html_base_dir, temp_folder=temp_folder, input_folder=input_folder, mean=mean, std=std), port)


def get_server(model, hooks, input_folder='./', input_size=[1, 3, 32, 32], classes=None, top=5, temp_folder='./tmp', port=5000, html_base_dir=None, mean=None, std=None):
    if platform.system() is 'Windows':
        temp_folder = '.\\tmp'
        os.system('mkdir %s' % temp_folder)
    else:
        os.system('mkdir -p %s' % temp_folder)

    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    # print(html_base_dir)
    validate_launch(html_base_dir)

    app = get_app(model, hooks, classes, top, input_size=input_size, html_base_dir=html_base_dir, temp_folder=temp_folder, input_folder=input_folder, mean=mean, std=std)
    http_server = WSGIServer(('', port), app, handler_class=Handler)
    http_server.real_started = False

    return http_server


class Handler(WSGIHandler):
    def log_request(self):
        self.server.real_started = True
        self.server.log.write(self.format_request() + '\n')