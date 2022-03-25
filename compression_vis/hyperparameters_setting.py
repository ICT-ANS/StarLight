import sys, os, time
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QMainWindow, QSlider, QLabel, QMessageBox, QFileDialog, QHBoxLayout, QPushButton, QApplication, QLineEdit, QDialog)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from collections import OrderedDict

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
from ui.qtui.compression.hyperparameters_setting_ui import * 

import numpy as np
import pynvml
import yaml

pynvml.nvmlInit()
#
from _init_paths import C


# 超参输入界面
class HyperparametersSettingWindow(QDialog):
    def __init__(self, dataset, model, prune_method, quan_method, is_online, is_offline):
        QDialog.__init__(self)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.dataset = dataset
        self.model = model
        self.prune_method = prune_method
        self.quan_method = quan_method
        self.is_online = is_online
        self.is_offline = is_offline

        self.set_enable()
        self.reset()
        self.ui.pushButton_input_path.clicked.connect(lambda: self.select_path(self.ui.lineEdit_input_path))
        self.ui.pushButton_output_path.clicked.connect(lambda: self.select_path(self.ui.lineEdit_output_path))
        self.ui.pushButton_dataset_path.clicked.connect(lambda: self.select_path(self.ui.lineEdit_dataset_path))
        self.ui.pushButton_confirm.clicked.connect(self.ok)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_quit.clicked.connect(self.quit)

        self.data = self.get_dict_data()

    def set_enable(self):
        if self.is_offline:
            # self.ui.lineEdit_output_path.setEnabled(False)
            # self.ui.lineEdit_input_path.setEnabled(False)
            # self.ui.lineEdit_dataset_path.setEnabled(False)
            # self.ui.pushButton_output_path.setEnabled(False)
            # self.ui.pushButton_input_path.setEnabled(False)
            # self.ui.pushButton_dataset_path.setEnabled(False)
            self.ui.lineEdit_gpus.setEnabled(False)
            self.ui.lineEdit_ft_bs.setEnabled(False)
            self.ui.lineEdit_ft_epochs.setEnabled(False)
            self.ui.lineEdit_ft_lr.setEnabled(False)
            self.ui.lineEdit_prune_sparisity.setEnabled(False)
        if self.prune_method == 'null':
            self.ui.lineEdit_ft_bs.setEnabled(False)
            self.ui.lineEdit_ft_epochs.setEnabled(False)
            self.ui.lineEdit_ft_lr.setEnabled(False)
            self.ui.lineEdit_prune_sparisity.setEnabled(False)

    def select_path(self, lineEdit):
        if os.path.exists('data/compression/outputs'):
            path = 'data/compression/outputs'
        else:
            path = './'
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", path)
        lineEdit.setText(path)


    def ok(self):
        output_path = self.ui.lineEdit_output_path.text()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        valid, info = self.valid_check()
        if valid:
            self.data = self.get_dict_data()
            self.accept()
        else:
            msg_box = QMessageBox(QMessageBox.Warning, "Alert", info)
            msg_box.exec_()

    def reset(self):
        currentfolder = os.path.abspath(os.path.dirname(__file__))
        hp_config_file = os.path.join(os.path.dirname(currentfolder), "compression_vis", "config", "hyperparameters_setting.yaml")
        with open(hp_config_file) as f:
            hp_config = yaml.safe_load(f)
        
        ft_bs = str(hp_config['default_setting']['{},{}'.format(self.dataset, self.model)]['ft_bs'])
        ft_epochs = str(hp_config['default_setting']['{},{}'.format(self.dataset, self.model)]['ft_epochs'])
        ft_lr = str(hp_config['default_setting']['{},{}'.format(self.dataset, self.model)]['ft_lr'])
        prune_sparisity = str(hp_config['default_setting']['{},{}'.format(self.dataset, self.model)]['prune_sparisity'])
        gpus = str(hp_config['default_setting']['{},{}'.format(self.dataset, self.model)]['gpus'])
        self.ui.lineEdit_ft_bs.setText(ft_bs)
        self.ui.lineEdit_ft_epochs.setText(ft_epochs)
        self.ui.lineEdit_ft_lr.setText(ft_lr)
        self.ui.lineEdit_prune_sparisity.setText(prune_sparisity)
        self.ui.lineEdit_gpus.setText(gpus)

        cache_folder = C.cache_dir
        output_path = os.path.join(cache_folder, 'outputs', '{}-{}/{}-{}-{}'.format(self.dataset, self.model, "online" if self.is_online else "offline", self.prune_method, self.quan_method))
        input_path = os.path.join(cache_folder, 'inputs', '{}-{}'.format(self.dataset, self.model))
        dataset_path = os.path.join(cache_folder, 'dataset', '{}'.format(self.dataset))
        # if not os.path.exists(cache_folder):
        #     os.makedirs(cache_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # if not os.path.exists(input_path):
        #     msg_box = QMessageBox(QMessageBox.Warning, "路径不存在", "请重新输入input path")
        #     msg_box.exec_()
        # if not os.path.exists(dataset_path):
        #     msg_box = QMessageBox(QMessageBox.Warning, "路径不存在", "请重新输入dataset path")
        #     msg_box.exec_()

        self.ui.lineEdit_output_path.setText(output_path)
        self.ui.lineEdit_input_path.setText(input_path)
        self.ui.lineEdit_dataset_path.setText(dataset_path)

    def quit(self):
        self.reject()

    def get_data(self):
        data = []
        data.append('超参数设置:')
        data.append('finetune learning rate：' + self.ui.lineEdit_ft_lr.text())
        data.append('finetune batch size：' + self.ui.lineEdit_ft_bs.text())
        data.append('finetune epochs：' + self.ui.lineEdit_ft_epochs.text())
        data.append('sparisity：' + self.ui.lineEdit_prune_sparisity.text())
        data.append('gpus：' + self.ui.lineEdit_gpus.text())
        data.append('output path：' + self.ui.lineEdit_output_path.text())
        data.append('input path：' + self.ui.lineEdit_input_path.text())
        data.append('dataset path：' + self.ui.lineEdit_dataset_path.text())
        return data

    def get_dict_data(self):
        data = {}
        # base
        data['ft_lr'] = float(self.ui.lineEdit_ft_lr.text())
        data['ft_bs'] = int(self.ui.lineEdit_ft_bs.text())
        data['ft_epochs'] = int(self.ui.lineEdit_ft_epochs.text())
        data['sparsity'] = float(self.ui.lineEdit_prune_sparisity.text())

        data['gpus'] = self.ui.lineEdit_gpus.text()
        try:
            data['gpus'] = int(data['gpus'])
        except:
            data['gpus'] = str(data['gpus'])
        data['output_path'] = self.ui.lineEdit_output_path.text()
        data['input_path'] = self.ui.lineEdit_input_path.text()
        data['dataset_path'] = self.ui.lineEdit_dataset_path.text()

        return data

    def valid_check(self):
        def is_float(data):
            try:
                float(data)
            except ValueError:
                return False
            return True

        def is_int(data):
            try:
                int(data)
            except ValueError:
                return False
            return True

        def is_in(data, minn=None, maxx=None, min_bound=True, max_bound=True):
            if minn is not None:
                if data < minn or (not min_bound and data == minn):
                    return False
            if maxx is not None:
                if data > maxx or (not max_bound and data == maxx):
                    return False
            return True

        def is_path(path):
            return os.path.exists(path)

        def check_gpus(gpus):
            gpus = gpus.split(', ')
            for gpu in gpus:
                if not is_int(gpu):
                    return False
                try:
                    pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
                except:
                    return False
            return True

        key = self.ui.label_ft_lr.text()
        text = self.ui.lineEdit_ft_lr.text()
        if not is_float(text):
            return False, '{}应输入实数'.format(key)
        if not is_in(float(text), 0., min_bound=False):
            return False, '{}应大于0'.format(key)

        key = self.ui.label_ft_bs.text()
        text = self.ui.lineEdit_ft_bs.text()
        if not is_int(text):
            return False, '{}应输入整数'.format(key)
        if not is_in(int(text), 0., min_bound=False):
            return False, '{}应大于0'.format(key)


        key = self.ui.label_ft_epochs.text()
        text = self.ui.lineEdit_ft_epochs.text()
        if not is_int(text):
            return False, '{}应输入整数'.format(key)
        if not is_in(int(text), 0, min_bound=False):
            return False, '{}应大于0'.format(key)

        key = self.ui.label_gpus.text()
        text = self.ui.lineEdit_gpus.text()
        if not check_gpus(text):
            return False, 'gpu不存在或输入不合法【输入格式： 0,1,2（多gpu） 或 0（单gpu）】'

        key = self.ui.label_output_path.text()
        text = self.ui.lineEdit_output_path.text()
        if not is_path(text):
            return False, '{}路径不存在'.format(key)

        key = self.ui.label_input_path.text()
        text = self.ui.lineEdit_input_path.text()
        if not is_path(text):
            return False, '{}路径不存在'.format(key)

        key = self.ui.label_dataset_path.text()
        text = self.ui.lineEdit_dataset_path.text()
        if not is_path(text):
            return False, '{}路径不存在'.format(key)

        return True, 'ok'