# -*- coding: UTF-8 -*-
import sys, os, re
rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(rootpath)
sys.path.append(os.path.join(os.path.dirname(__file__), "tmp/"))

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np

import time
import shutil
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QMainWindow, QSlider, QLabel, QHBoxLayout, QPushButton, QApplication, QLineEdit, QDialog, QGraphicsPixmapItem, QGraphicsScene, QMessageBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView

from _init_paths import C

import ui.echarts.utils as echarts

from ui.qtui.compression.compression_ui import * 
from compression_vis.hyperparameters_setting import HyperparametersSettingWindow
from algorithms import *

import torch

import cgitb
import inspect
import ctypes
import threading
import copy
import json
import yaml
cgitb.enable(format='text')

from torchvision import models
import psutil
import subprocess

sys.path.append(os.path.join(C.work_dir, "algorithms/compression/nets/VGG_SSD"))
from algorithms.compression.nets.VGG_SSD.models.model_builder import SSD
from ui.quiver.utils import ModelViewer

sys.path.append(os.path.join(C.work_dir, 'algorithms/compression/nets/PSPNet/models'))
sys.path.append(os.path.join(C.work_dir, 'algorithms/compression/nets/DeepLabV3Plus'))



sub_process = None

def cvtPath(path):
    return path.replace(os.sep, "/")

class CompressVisTask(QThread):
    signalUpdateUi = pyqtSignal(bool, dict, dict, str, str)  # 定义更新UI信号: is_finish, baseline_dict, method_dict, error_str, output_file

    def __init__(self, shell_path, var_names, args):
        """Init CompressVisTask class

        Parameters
        ----------
        shell_path : str
            the shell path for compressing
        var_names : list
            metrics for evaluting network, e.g. accuracy, FLOPs, Parameters
        args : dict
            args should contain follows:
                dataset: str
                model: model
                is_online: bool
                is_offline: bool
                prune_method: str
                quan_method: str
                ft_lr: float
                ft_bs: int
                ft_epochs: int
                sparsity: float
                prune_epochs: int
                gpus: str
                output_path: str
                intput_path: str
                dataset_path: str
        """        
        super().__init__()
        self.shell_path = shell_path
        self.var_names = var_names
        self.args = args

    def run_algo(self):
        """run algorithm by shell
        """        
        global sub_process
        command = 'bash {}'.format(self.shell_path) + \
                    ' {}'.format(self.args['dataset']) + \
                    ' {}'.format(self.args['model']) + \
                    ' {}'.format(self.args['prune_method']) + \
                    ' {}'.format(self.args['quan_method']) + \
                    ' {}'.format(self.args['ft_lr']) + \
                    ' {}'.format(self.args['ft_bs']) + \
                    ' {}'.format(self.args['ft_epochs']) + \
                    ' {}'.format(self.args['sparsity']) + \
                    ' {}'.format(self.args['gpus']) + \
                    ' {}'.format(self.args['input_path']) + \
                    ' {}'.format(self.args['output_path']) + \
                    ' {}'.format(self.args['dataset_path'])
        # print(command)
        sub_process = subprocess.Popen(command, shell=True)


    def run(self):
        """run online mode or offline mode
        """        
        if self.args['is_online']:
            self.online()
        elif self.args['is_offline']:
            self.offline()

    def online(self):
        """online mode
        """        
        thread = threading.Thread(target=self.run_algo)
        thread.start()

        self.sleep(3)

        logs_file = os.path.join(self.args['output_path'], 'logs.yaml')

        global sub_process

        while not os.path.exists(logs_file): # 每隔 1s 检测是否生成 log_file, 未生成则检查程序是否成功运行
            if not (sub_process and psutil.pid_exists(sub_process.pid)): # 程序未运行
                self.signalUpdateUi.emit(False, {}, {}, 'The program did not run successfully.', '')
                return
            
            self.sleep(1)
        
        with open(logs_file) as f:
            log_config = yaml.safe_load(f)

        uiBaseline = {}
        uiInitMethod = {}
        uiMethod = {}
        for key in self.var_names:
            uiBaseline[key] = log_config[key]['baseline']
            uiInitMethod[key] = 0
            uiMethod[key] = log_config[key]['method']


        if not os.path.exists(logs_file):
            self.signalUpdateUi.emit(False, {}, {}, 'File of {:s} does not exist'.format(logs_file), '')
            return

        with open(logs_file) as f:
            log_config = yaml.safe_load(f)

        uiBaseline = {}
        uiInitMethod = {}
        for key in self.var_names:
            uiBaseline[key] = log_config[key]['baseline']
            uiInitMethod[key] = 0
            uiMethod[key] = log_config[key]['method']
        
        self.signalUpdateUi.emit(False, uiBaseline, uiInitMethod, '', '')

        while True: # 每隔 10 s 检查程序是否运行完成，如果运行完成，则返回
            is_finished = True
            if not os.path.exists(logs_file):
                self.sleep(1)
                continue
            
            with open(logs_file) as f:
                log_config = yaml.safe_load(f)
            for key in self.var_names: # 检查程序是否运行完成
                if log_config[key]['method'] == None:
                    is_finished = False
                    break
            if 'Output_file' not in log_config:  # 检查程序是否运行完成
                is_finished = False
            
            if is_finished: # 如果运行完成，则返回
                output_file = log_config['Output_file']

                uiBaseline = {}
                uiMethod = {}
                for key in self.var_names:
                    uiBaseline[key] = log_config[key]['baseline']
                    uiMethod[key] = log_config[key]['method']
                self.signalUpdateUi.emit(True, uiBaseline, uiMethod, '', output_file)

                return
            
            self.sleep(10) # 睡眠 10s

    def offline(self):
        """offline mode
        """        
        logs_file = os.path.join(self.args['output_path'], 'logs.yaml')

        if not os.path.exists(logs_file):
            self.signalUpdateUi.emit(False, {}, {}, 'File of {:s} does not exist'.format(logs_file), '')
            return

        with open(logs_file) as f:
            log_config = yaml.safe_load(f)

        uiBaseline = {}
        uiMethod = {}
        for key in self.var_names:
            uiBaseline[key] = log_config[key]['baseline']
            uiMethod[key] = log_config[key]['method']


        output_file = log_config['Output_file']

        self.signalUpdateUi.emit(True, uiBaseline, uiMethod, '', output_file)


# 主界面
class MainWindow(QMainWindow):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        """Init QMainWindow class

        Parameters
        ----------
        parent : optional
            default None
        flags : optional
            default Qt.WindowFlags()
        """        
        super(MainWindow, self).__init__(parent=parent, flags=flags)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initConfig()

        # 确认与重置按钮
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_reset.setEnabled(False)
        self.ui.pushButton_start.clicked.connect(self.compressStartBtnCallback)
        self.ui.pushButton_reset.clicked.connect(self.resetStartBtnCallback)

        # 在线与离线
        self.ui.checkBox_3.clicked.connect(self.offlineClickedCallback) # 离线
        self.ui.checkBox_4.clicked.connect(self.onlineClickedCallback) # 在线

        self.setPerformance()
        self.ui.comboBox.currentIndexChanged.connect(self.setPerformance)
        self.ui.comboBox.currentIndexChanged.connect(self.setMethod)

        self.ui.pushButton_2.clicked.connect(self.setHyperparameters)
        self.ui.pushButton_modelvis.clicked.connect(self.modelVisBtnCallback)
        self.ui.pushButton_compmodelvis.clicked.connect(self.prunedModelVisBtnCallback)
        self.ui.tabWidget.tabBarClicked.connect(self.tabWidgetClickedCallback)

        self.modelVis = ModelViewer(self.ui.scrollArea_7)

        self.ui.comboBox.currentIndexChanged.connect(self.setHtml)
        # self.initHtml()
        # self.initHtml()

        # self.currentMethod = 'null'
        self.train = False

        self.epochTimer = QTimer()
        self.epochTimer.timeout.connect(self.updateEpochDisplay)
        self.epochTimer.start(500)
        self.display = False
        self.timerCount = 0

        self.process_running = False

        # move to center
        desktop = QApplication.desktop()
        ox, oy = (desktop.width() - self.width()) // 2, (desktop.height() - self.height()) // 2
        self.move(ox, oy)

        self.showMaximized()
        self.show()  # QMainWindow必须调用该函数，但QDialog不用
    
    # def slotRezise(self):
    #     self.resizeEvent(None)

    def initConfig(self):
        """read config file and init
        """        
        currentfolder = os.path.abspath(os.path.dirname(__file__))
        global_config_file = os.path.join(os.path.dirname(currentfolder), "compression_vis/config", "global.yaml")
        with open(global_config_file) as f:
            global_config = yaml.safe_load(f)
        # dataset and model
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(global_config['dataset,model'])
        self.ui.comboBox.setCurrentIndex(global_config['dataset,model'].index(global_config['default']['dataset,model']))
        # is_online/is_offline
        self.ui.checkBox_4.setChecked(global_config['default']['is_online']) # 在线
        self.ui.checkBox_3.setChecked(global_config['default']['is_offline']) # 离线
        # prune
        self.ui.comboBox_4.clear()
        self.ui.comboBox_4.addItems(global_config['prune_method'])
        self.ui.comboBox_4.setCurrentIndex(global_config['prune_method'].index(global_config['default']['prune_method']))
        # quan
        self.ui.comboBox_5.clear()
        self.ui.comboBox_5.addItems(global_config['quan_method'])
        self.ui.comboBox_5.setCurrentIndex(global_config['quan_method'].index(global_config['default']['quan_method']))

        self.global_config = global_config

    def initHtml(self):
        """init Html for bar figure
        """        
        currentfolder = os.path.abspath(os.path.dirname(__file__))
        root = os.path.join(os.path.dirname(currentfolder), "ui/echarts", "html")
        lineHtmlfile = os.path.join(root, "line.html")
        barHtmlfile = os.path.join(root, "bar.html")
        pieHtmlfile = os.path.join(root, "pie.html")

        # 存储损失柱状图
        self.storageBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_6, xAxis=[], legends=["baseline storage/MB", "method storage/MB"], colors=['#FB3207', '#0780FB'], title='Storage对比') 

        # acc柱状图
        self.accBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea, xAxis=[], legends=["baseline accuracy/%", "method accuracy/%"], colors=['#FB3207', '#0780FB'], title='Accuracy对比') 

        # flops柱状图
        self.flopsBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_2, xAxis=[], legends=["baseline FLOPs/M", "method FLOPs/M"], colors=['#FB3207', '#0780FB'], title='FLOPs对比') 

        # params柱状图
        self.paramsBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_5, xAxis=[], legends=["baseline parameters/M", "method parameters/M"], colors=['#FB3207', '#0780FB'], title='Parameters对比') 

        # 推理时间柱状图
        self.infertimeBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_4, xAxis=[], legends=["baseline infer_time/ms", "method infer_time/ms"], colors=['#FB3207', '#0780FB'], title='Infer Time对比') 

    def setHtml(self):
        """set Html for bar figure
        """        
        key = '{},{}'.format(self.getData('dataset'), self.getData('model'))
        if key not in self.global_config['figures']:
            return
        
        if not self.ui.pushButton_start.isEnabled(): # 如果程序跑完未重置，不修改
            return 

        currentfolder = os.path.abspath(os.path.dirname(__file__))
        root = os.path.join(os.path.dirname(currentfolder), "ui/echarts", "html")
        barHtmlfile = os.path.join(root, "bar.html")

        titles = self.global_config['figures'][key]['titles']
        baseline_legends = self.global_config['figures'][key]['baseline_legends']
        method_legends = self.global_config['figures'][key]['method_legends']
        baseline_colors = self.global_config['figures'][key]['baseline_colors']
        method_colors = self.global_config['figures'][key]['method_colors']
        
        # top_center, top_right, down_left, down_center, down_right: acc柱状图, flops柱状图, 推理时间柱状图, params柱状图, 存储损失柱状图
        self.accBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea, xAxis=[], legends=[baseline_legends[0], method_legends[0]], colors=[baseline_colors[0], method_colors[0]], title=titles[0]) 
        self.flopsBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_2, xAxis=[], legends=[baseline_legends[1], method_legends[1]], colors=[baseline_colors[1], method_colors[1]], title=titles[1]) 
        self.infertimeBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_4, xAxis=[], legends=[baseline_legends[2], method_legends[2]], colors=[baseline_colors[2], method_colors[2]], title=titles[2]) 
        self.paramsBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_5, xAxis=[], legends=[baseline_legends[3], method_legends[3]], colors=[baseline_colors[3], method_colors[3]], title=titles[3]) 
        self.storageBarHtml = echarts.Bar(cvtPath(barHtmlfile), self.ui.scrollArea_6, xAxis=[], legends=[baseline_legends[4], method_legends[4]], colors=[baseline_colors[4], method_colors[4]], title=titles[4]) 

    def updateEpochDisplay(self):
        """update data each epoch
        """    
        # self.resizeEvent(None)

        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)  #设置字体颜色
        self.ui.label_7.setPalette(pe)

        # global sub_process
        # if sub_process:
        #     sub_process.poll()

        if self.getData('is_offline'):
            self.ui.label_7.setText('Offline Mode')
        elif self.getData('is_online'):
            self.ui.label_7.setText('Online Mode')
            if self.display:
                if self.timerCount % 3 == 0:
                    strs = '...'
                elif self.timerCount % 3 == 1:
                    strs = '......'
                elif self.timerCount % 3 == 2:
                    strs = '.........'

                self.ui.label_7.setText('Online mode and the program is running: {}'.format(strs))
                self.timerCount += 1
            else:
                self.timerCount = 0

    def getData(self, type):
        """_summary_

        Parameters
        ----------
        type : str
            dataset | model | prune_method | quan_method | is_online | is_offline

        Returns
        -------
        str
            str of data
        """        
        if type == 'dataset':
            if len(self.ui.comboBox.currentText().split(',')) == 1:
                return 'null' # 当self.ui.comboBox.clear()是，其内容会为‘’，导致调用setPerformanc并报错
            return self.ui.comboBox.currentText().split(',')[0]
        elif type == 'model':
            if len(self.ui.comboBox.currentText().split(',')) == 1:
                return 'null'
            return self.ui.comboBox.currentText().split(',')[1]
        elif type == 'prune_method':
            return self.ui.comboBox_4.currentText()
        elif type == 'quan_method':
            return self.ui.comboBox_5.currentText()
        elif type == 'is_online':
            return self.ui.checkBox_4.isChecked()
        elif type == 'is_offline':
            return self.ui.checkBox_3.isChecked()
    
    def checkData(self):
        """_summary_

        Returns
        -------
        bool
            whether the input is valid
        """        
        if self.getData('dataset') == 'null' or self.getData('model') == 'null': # 如果数据集和模型为null，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select a dataset and a network')
            # QMessageBox.about(self, "Message", 'Please select a dataset and a network')
            return False
        if self.getData('is_online') is False and self.getData('is_offline') is False: # 如果在线离线模式为空，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select online or offline mode')
            # QMessageBox.about(self, "Message", 'Please select online or offline mode”')
            return False
        if self.getData('prune_method') == 'null' and self.getData('quan_method') == 'null': # 如果剪枝为null且量化为null，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select a pruning or quantization method')
            # QMessageBox.about(self, "Message", 'Please select a pruning or quantization method')
            return False

        dataset = self.getData('dataset')
        model = self.getData('model')
        prune_method = self.getData('prune_method')
        quan_method = self.getData('quan_method')
        dataset_model_key = '{},{}'.format(dataset, model)
        method_key = '{},{}'.format(prune_method, quan_method)
        if not dataset_model_key in self.global_config['support_combinations']: # 如果数据集、模型不在支持的组合内，弹出提示框
            text = 'Please select one of the (dataset, network) pairs: \n'
            for key in self.global_config['support_combinations']:
                text += '  ({})\n'.format(key)
            self.jumpQMessageBox("Message", text)
            # QMessageBox.about(self, "Message", text)
            return False
        if not method_key in self.global_config['support_combinations'][dataset_model_key]:  # 如果剪枝、量化方法不在支持的组合内，弹出提示框
            text = 'Please select one of the (pruning, quantization) methods: \n'
            for key in self.global_config['support_combinations'][dataset_model_key]:
                text += '  ({})\n'.format(key)
            self.jumpQMessageBox("Message", text)
            # QMessageBox.about(self, "Message", text)
            return False
        if dataset in ['Cityscapes'] and self.getData('is_online'):
            self.jumpQMessageBox("Message", 'This method does support online mode due to GPU memory limitations')
            return False

        return True
    
    def checkMethod(self):
        dataset = self.getData('dataset')
        model = self.getData('model')
        prune_method = self.getData('prune_method')
        quan_method = self.getData('quan_method')
        dataset_model_key = '{},{}'.format(dataset, model)
        method_key = '{},{}'.format(prune_method, quan_method)
        if dataset != 'null' and model != 'null':
            if prune_method != '' or quan_method != '':
                if prune_method != 'null' or quan_method != 'null':
                    if not method_key in self.global_config['support_combinations'][dataset_model_key]:  # 如果剪枝、量化方法不在支持的组合内，弹出提示框
                        text = 'Please select one of the (pruning, quantization) methods: \n'
                        for key in self.global_config['support_combinations'][dataset_model_key]:
                            text += '  ({})\n'.format(key)
                        self.jumpQMessageBox("Message", text)
                        self.ui.comboBox_4.setCurrentIndex(self.ui.comboBox_4.findText('null'))
                        self.ui.comboBox_5.setCurrentIndex(self.ui.comboBox_5.findText('null'))

    def setMethod(self):
        """set default value
        """        
        dataset = self.getData('dataset')
        model = self.getData('model')
        dataset_model_key = '{},{}'.format(dataset, model)
        if dataset_model_key == "null,null":
            method_list = []
        else:
            method_list = self.global_config['support_combinations'][dataset_model_key]
        prune_list = ['null']
        quan_list = ['null']
        for m in method_list:
            prune_m, quan_m = m.split(',')
            if prune_m not in prune_list:
                prune_list.append(prune_m)
            if quan_m not in quan_list:
                quan_list.append(quan_m)

        if hasattr(self, 'connect_45'):#
            self.ui.comboBox_4.currentIndexChanged.disconnect(self.checkMethod)
            self.ui.comboBox_5.currentIndexChanged.disconnect(self.checkMethod)
        
        # prune
        self.ui.comboBox_4.clear()
        self.ui.comboBox_4.addItems(prune_list)
        self.ui.comboBox_4.setCurrentIndex(prune_list.index('null'))
        
        # quan
        self.ui.comboBox_5.clear()
        self.ui.comboBox_5.addItems(quan_list)
        self.ui.comboBox_5.setCurrentIndex(quan_list.index('null'))
        

        self.ui.comboBox_4.currentIndexChanged.connect(self.checkMethod) # check method valid
        self.ui.comboBox_5.currentIndexChanged.connect(self.checkMethod) # check method valid
        self.connect_45 = None

    def setPerformance(self):
        """set Performance based on config file
        """        
        # currentfolder = os.path.abspath(os.path.dirname(__file__))
        # global_config_file = os.path.join(os.path.dirname(currentfolder), "compression_vis/config", "global.yaml")
        # with open(global_config_file) as f:
        #     global_config = yaml.safe_load(f)
        # dataset = self.getData('dataset')
        # model = self.getData('model')
        # if '{},{}'.format(dataset, model) in global_config['origin_performance']:
        #     performance = global_config['origin_performance']['{},{}'.format(dataset, model)]
        #     text = ''
        #     for key in performance:
        #         text += '{}: {}\n'.format(key, performance[key])
        #     # text = 'Acc: {}\nFLOPs: {}\nParams: {}\n'.format(performance['acc'], performance['flops'], performance['params'])
        # else:
        #     text = ''

        # self.ui.label_3.setText(text) 

        self.ui.label_3.setText('')

    def setHyperparameters(self):
        """set Hyperparameters 
        """        
        if not self.checkData():
            return
        dialog = HyperparametersSettingWindow(self.getData('dataset'), self.getData('model'), self.getData('prune_method'), self.getData('quan_method'), self.getData('is_online'), self.getData('is_offline'))
        if dialog.exec_(): # if self.getData('is_offline') or dialog.exec_():
            data = dialog.get_data()
            datastr = ''
            for d in data:
                datastr += d + '\n'
            self.ui.textBrowser.setText(datastr)
            self.dialog = dialog
            return True
        else:
            return False

    def get_online_pid(self, cmd_key):
        process_list =  list(psutil.process_iter())
        regex = "pid=(\d+),\sname=\'" + "python" + "\'"
        ini_regex = re.compile(regex)

        for p in process_list:
            process_info = str(p)

            result = ini_regex.search(process_info)

            if result != None:
                argv = p.cmdline()
                for arg in argv:
                    if cmd_key in arg:
                        return p.pid
        return None

    def kill_online(self, pid):
        choice = QMessageBox.No
        p = psutil.Process(pid)
        choice = QMessageBox.question(self, "Warning", f'Do you want to stop the online training? [PID: {pid}]', QMessageBox.Yes | QMessageBox.No)  # 1
        if choice == QMessageBox.Yes:
            for child in p.children(recursive=True):
                print("kill: ", child)
                child.kill()
            print("kill: ", p)
            p.kill()
            return True #prcoess killed
        return False #

    def jumpQMessageBox(self, title, message):
        """jump QMessageBox

        Parameters
        ----------
        title : str
            QMessageBox title
        message : str
            QMessageBox message
        """        
        # box = QMessageBox(QMessageBox.Warning, title, message)
        # box.addButton("确定", QMessageBox.YesRole)
        # #self.box.addButton(self.tr("取消"), QMessageBox.NoRole)
        # box.exec_()
        QMessageBox.information(self, title, message)

    def getHyperparameters(self):
        """get Hyperparameters

        Returns
        -------
        dict
            Hyperparameters
        """        
        return self.dialog.data

    def tabWidgetClickedCallback(self):
        """Callback when clicking tabWidget
        """        
        self.resizeEvent(None)

    def onlineClickedCallback(self):
        """Callback when clicking online
        """        
        self.ui.checkBox_3.setChecked(not self.ui.checkBox_4.isChecked())
        self.ui.pushButton_2.setEnabled(True)

    def offlineClickedCallback(self):
        """Callback when clicking offline
        """        
        self.ui.checkBox_4.setChecked(not self.ui.checkBox_3.isChecked())
        self.ui.pushButton_2.setEnabled(True)

    def closeEvent(self, event):  # 函数名固定不可变
        """check whether the process is finished
        """   
        pid = None
        if self.getData('is_online'):   
            pid = self.get_online_pid(self.getData('model'))
            if pid is not None and not self.kill_online(pid):
                event.ignore() #保留窗口
                return
        event.accept()  # 关闭窗口

    # 压缩确认按钮回调函数
    def compressStartBtnCallback(self):
        """Callback when clicking compressStart Butten
        """        
        # 初始化
        self.resizeEvent(None)
        self.ui.label_17.setText('') # 压缩输出

        # 逻辑检查
        if not self.checkData():
            return
        if self.ui.textBrowser.toPlainText() == '': # 如果超参数文本框为空
            flag = self.setHyperparameters()
            if not flag:
                return 
            # if self.getData('is_online'): # 如果是在线模式，弹出超参数设置框
            #     self.setHyperparameters()
            # else: # 如果是离线模式，使用默认值设置超参数文本框
            #     dialog = HyperparametersSettingWindow(self.getData('dataset'), self.getData('model'), self.getData('prune_method'), self.getData('quan_method'), self.getData('is_online'), self.getData('is_offline'))
            #     data = dialog.get_data()
            #     datastr = ''
            #     for d in data:
            #         datastr += d + '\n'jjlllljjjjjjdjflajlsjlfdkfjkdjfkd
            #     self.ui.textBrowser.setText(datastr)
        
        if self.ui.textBrowser.toPlainText() == '' and self.getData('is_online'): # 如果在线模式输出超参数时直接关闭窗口，返回
            return 
        
        self.storageBarHtml.build()
        self.accBarHtml.build()
        self.flopsBarHtml.build()
        self.paramsBarHtml.build()
        self.infertimeBarHtml.build()

        self.ui.label_7.setText('The program is running, please wait.')
        self.process_running = True

        # if self.getData('is_offline'):
        #     self.setHyperparameters()

        oldlogfile = os.path.join(self.getHyperparameters()['output_path'],'logs.yaml')
        if self.getData('is_online') and os.path.exists(oldlogfile):
            # QtWidgets.QMessageBox.about(self, '提示', '该路径下有其他压缩程序保存的文件，请选择另一个空文件夹运行在线压缩程序')
            # self.jumpQMessageBox('提示', '该路径下有其他压缩程序保存的文件，请选择另一个空文件夹运行在线压缩程序')
            choice = QMessageBox.question(self, "Message", f'Files exist in this path, do you want to delete them?',  
                        QMessageBox.Yes | QMessageBox.No)  # 1
            if choice == QMessageBox.Yes:
                # shutil.rmtree(self.getHyperparameters()['output_path'])
                os.remove(oldlogfile)
            else:
                return

        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_reset.setEnabled(True)

        args = copy.deepcopy(self.getHyperparameters())
        args['dataset'] = self.getData('dataset')
        args['model'] = self.getData('model')
        args['is_online'] = self.getData('is_online')
        args['is_offline'] = self.getData('is_offline')
        args['prune_method'] = self.getData('prune_method')
        args['quan_method'] = self.getData('quan_method')
        key = '{},{}'.format(self.getData('dataset'), self.getData('model'))
        self.compressVisTask = CompressVisTask(self.global_config['shell_path'][key], self.global_config['figures'][key]['var_names'], args)
        self.compressVisTask.signalUpdateUi.connect(self.updateUiCompressCallback)
        self.compressVisTask.start()
        self.display = True
    
    # 重置按钮回调函数
    def resetStartBtnCallback(self):
        """Callback when clicking reset Butten
        """        
        pid = None
        if self.getData('is_online'):   
            pid = self.get_online_pid(self.getData('model'))
            if pid is not None and not self.kill_online(pid):
                return 
        
        # 把数据集、模型、在线/离线模式和压缩方法变成缺省值，清空在线离线模式，超参数字图，清空柱状图
        self.initConfig()
        self.ui.textBrowser.setText('')
        self.display = False
        self.ui.label_7.setText('')
        self.ui.label_17.setText('')
        self.resetHtml()

        # “重置”按钮变灰，此时“确认”按钮应该是从灰色变成激活状态
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_reset.setEnabled(False)
    

    def modelVisBtnCallback(self):
        """Callback when clicking modelVis Butten
        """       
        dataset = self.getData('dataset')
        model = self.getData('model')
        datapath = os.path.join(C.quiver_dir, dataset)
        model_path = os.path.join(C.model_vis, '{}-{}'.format(dataset, model), 'model.pth')
        if dataset == 'null':
            self.jumpQMessageBox("Message", 'Please select a dataset and a network')
            return
        if not os.path.exists(model_path):
            self.jumpQMessageBox("Message", 'The model does not exist: {}'.format(model_path))
            # QMessageBox.about(self, "Message", )
            return
        model = torch.load(model_path).cpu().eval()
        # cnt = 0
        # for m in model.modules():
        #     cnt += 1
        # print('count:', cnt)
        self.modelVis.set_img_size(self.global_config['img_size']['{},{}'.format(self.getData('dataset'), self.getData('model'))])
        self.modelVis.slotUpdateModel(model, datapath)
        self.resizeEvent(None) 
    
    def prunedModelVisBtnCallback(self):
        """Callback when clicking prunedModel Butten
        """       

        dataset = self.getData('dataset')
        model = self.getData('model')
        prune_method = self.getData('prune_method')
        quan_method = self.getData('quan_method')
        
        # if prune_method == 'null' or quan_method != 'null':
        #     # QMessageBox.about(self, "提示", '仅支持可视化剪枝后的压缩网络')
        #     self.jumpQMessageBox("提示", '仅支持可视化剪枝后的压缩网络')
        #     return
        # datapath = os.path.join(C.quiver_dir, dataset)
        # model_path = os.path.join(C.model_vis, '{}-{}'.format(dataset, model), '{}-{}.pth'.format("online" if self.getData('is_online') else "offline", prune_method))

        if prune_method == 'null':
            if quan_method != 'null':
                datapath = os.path.join(C.quiver_dir, dataset)
                model_path = os.path.join(C.model_vis, '{}-{}'.format(dataset, model), 'model.pth')
            else:
                self.jumpQMessageBox("Message", 'Please select a compression method')     
                return
        else:
            datapath = os.path.join(C.quiver_dir, dataset)
            model_path = os.path.join(C.model_vis, '{}-{}'.format(dataset, model), '{}-{}.pth'.format("online" if self.getData('is_online') else "offline", prune_method))

        if self.get_online_pid(self.getData('model')): #在线进程仍然存在
            # QtWidgets.QMessageBox.about(self, 'Online process running!', '程序正在运行，请结束后再可视化压缩后模型')
            self.jumpQMessageBox('Message', 'The program is running, please visualize the compressed model after it finishes')
            return
        if dataset == 'null':
            self.jumpQMessageBox("Message", 'Please select a dataset and a network')
            return
        if not os.path.exists(model_path):
            # QMessageBox.about(self, "提示", '模型不存在: {}'.format(model_path))
            self.jumpQMessageBox("Message", 'The model does not exist: {}'.format(model_path))
            return
        model = torch.load(model_path).cpu().eval()
        # cnt = 0
        # for m in model.modules():
        #     cnt += 1
        # print('count:', cnt)
        self.modelVis.set_img_size(self.global_config['img_size']['{},{}'.format(self.getData('dataset'), self.getData('model'))])
        self.modelVis.slotUpdateModel(model, datapath)
        self.resizeEvent(None) 

    def resizeEvent(self, a0):    
        num = self.ui.tabWidget.count()
        cur_index = self.ui.tabWidget.currentIndex()
        for i in range(num):
            self.ui.tabWidget.setCurrentIndex(i)
        self.ui.tabWidget.setCurrentIndex(cur_index)

        return super().resizeEvent(a0)

    def resetHtml(self):
        self.storageBarHtml.destroy()
        self.accBarHtml.destroy()
        self.flopsBarHtml.destroy()
        self.paramsBarHtml.destroy()
        self.infertimeBarHtml.destroy()
        self.modelVis.destroy()
        # self.modelVis = ModelViewer(self.ui.scrollArea_7)
        # self.ui.scrollArea_7.hide()

    def updateUiCompressCallback(self, is_finished, uiBaseline, uiMethod, error, output_file):
        """Callback when updating Compress UI

        Parameters
        ----------
        is_finished : bool
            whether the process is finished
        uiBaseline : dict
            accuracy, FLOPs, Params, Storage, Infer Time of Baseline
        uiMethod : dict
            accuracy, FLOPs, Params, Storage, Infer Time of Method
        error : str
            error message
        output_file : str
            output file path
        """        
        if error != '':
            # QMessageBox.about(self, "错误", '发生错误：{:s}， 请检查后重新运行'.format(error))
            self.jumpQMessageBox("Error", 'An error occurred: {:s}, please check and run again.'.format(error))
            self.resetStartBtnCallback()
            return

        # 动态更新方式，每次传输单个数据
        key = '{},{}'.format(self.getData('dataset'), self.getData('model'))
        var_names = self.global_config['figures'][key]['var_names']
        self.accBarHtml.update(['{:.2f}'.format(uiBaseline[var_names[0]]), '{:.2f}'.format(uiMethod[var_names[0]])])
        self.flopsBarHtml.update(['{:.2f}'.format(uiBaseline[var_names[1]]), '{:.2f}'.format(uiMethod[var_names[1]])])
        self.infertimeBarHtml.update(['{:.2f}'.format(uiBaseline[var_names[2]]), '{:.2f}'.format(uiMethod[var_names[2]])])
        self.paramsBarHtml.update(['{:.2f}'.format(uiBaseline[var_names[3]]), '{:.2f}'.format(uiMethod[var_names[3]])])
        self.storageBarHtml.update(['{:.2f}'.format(uiBaseline[var_names[4]]), '{:.2f}'.format(uiMethod[var_names[4]])])

        performance = self.global_config['origin_performance']['{},{}'.format(self.getData('dataset'), self.getData('model'))]
        text = ''
        for key in performance:
            text += '{}: {} {}\n'.format(key, uiBaseline[var_names[var_names.index(key)]], performance[key].split()[-1]) # 指标名称，数值，单位
        self.ui.label_3.setText(text)

        if is_finished:
            # 压缩输出
            performance = self.global_config['origin_performance']['{},{}'.format(self.getData('dataset'), self.getData('model'))]
            text = ''
            for key in performance:
                text += '{}: {} {}\n'.format(key, uiMethod[var_names[var_names.index(key)]], performance[key].split()[-1]) # 指标名称，数值，单位
            self.ui.label_17.setText(text)

            # 关闭display
            self.display = False

            # 弹出路径
            info = 'Path of the compressed network: {}'.format(output_file)
            # QMessageBox.about(self, "压缩结果", info)
            self.jumpQMessageBox("Results of compression", info)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainW = MainWindow()

    sys.exit(app.exec_())