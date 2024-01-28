import copy
from enum import EnumMeta
import subprocess
import sys, re
import time

from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem, QMessageBox)
from torch.utils.data import DataLoader

from config import C

from ui.qtui.nas.nas_ui import *
from ui.quiver.quiver_utils import ModelViewer

from nas_vis.nas_models import genotypes
from nas_vis.hyperparams_vis import HyperparametersSettingWindow
from nas_vis.nas_models.model import NetworkCIFAR as Network
from nas_vis.utils import *

import nas_vis.nas_burgerformer.net
from nas_vis.nas_burgerformer import arch
from nas_vis.nas_burgerformer.timm.models import create_model, resume_checkpoint
from nas_vis.nas_burgerformer.timm.data import create_loader, create_dataset

import cv2
import numpy as np
import psutil
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

# 用于数据交互的全局变量
counter = 0
uiData = 0
update_threshold = -1
sub_process = None
dataset = 'CIFAR-10'
CIFAR10_CLASS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
method_list = ['DARTS', 'GDAS']
method_index_list = ['lr', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'genotype', 'epoch']

data_dict = {}
for i, _m in enumerate(method_list):
    data_dict[_m] = from_darts_read_log(method=_m,
                                        log_name='{}/{}/logdir/Offline_{}_{}.log'.format(C.cache_dir, _m, _m, dataset),
                                        key_words=method_index_list)
# add burgerformer
data_dict['BurgerFormer'] = from_burgerformer_read_log(method='BurgerFormer',
                                            log_name='{}/{}/logdir/Offline_{}_{}.log'.format(C.cache_dir, 'BurgerFormer', 'BurgerFormer', 'ImageNet-100'),
                                            key_words=['acc', 'flops', 'params'])

print('Start visualization.')


# 任务线程
class NASVisTask(QThread):
    signalUpdateUi = pyqtSignal()  # 定义更新UI信号

    def __init__(self, args):
        """
        A dynamically changeable parameter args is declared.

        Parameters
        ----------
        args : list
            parameter
        """
        super().__init__()
        self.args = args

    def run(self):
        """
        Run function.

        Returns
        -------
        function
            call function
        """
        if self.args['is_online']:
            self.online()
        elif self.args['is_offline']:
            self.offline()
        else:
            raise ValueError('Wrong mode for NASVisTask.')

    def online(self):
        """
        This function runs the algorithm online.

        Returns
        -------
        None
        """

        while True:
            self.sleep(1)
            self.signalUpdateUi.emit()

    def offline(self):
        """
        This function runs the algorithm online.

        Returns
        -------
        None
        """
        offline_data = data_dict[self.args['method']]
        while True:
            if 'flops' in offline_data: # burgerformer
                global counter
                global update_threshold
                global acc_iter
                global flops_iter
                global params_iter

                self.sleep(1)

                if counter < update_threshold:
                    acc_iter = offline_data['acc'][counter]
                    flops_iter = offline_data['flops'][counter]
                    params_iter = offline_data['params'][counter]
                    counter += 1
                else:
                    break
                self.signalUpdateUi.emit()

            else:
                global lr_epoch
                global train_loss_epoch
                global train_acc_epoch
                global valid_loss_epoch
                global valid_acc_epoch

                self.sleep(1)
                # time.sleep(5)

                if counter < update_threshold:
                    lr_epoch = offline_data['lr'][counter]
                    train_loss_epoch = offline_data['train_loss'][counter]
                    train_acc_epoch = offline_data['train_acc'][counter]
                    valid_loss_epoch = offline_data['valid_loss'][counter]
                    valid_acc_epoch = offline_data['valid_acc'][counter]
                    counter += 1
                else:
                    # counter = update_threshold
                    break
                self.signalUpdateUi.emit()


# 主界面
class MainWindow(QMainWindow):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(MainWindow, self).__init__(parent=parent, flags=flags)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normal_scene = QGraphicsScene()
        self.reduce_scene = QGraphicsScene()
        self.infer_img_scene = QGraphicsScene()
        self.online = False
        self.start_offline = 0
        self.online_running = False
        self.online_killed = False
        self.online_switch = False

        self.ui = Ui_NAS()
        self.ui.setupUi(self)

        self.finish_timer = QTimer()
        self.finish_timer.timeout.connect(self.check_online_finished)

        # 初始化html曲线
        self.initConfig(first_init=True)

        # 所有按钮初始化
        self.ui.pushButton_start.setEnabled(True)  # 确定
        self.ui.pushButton_reset.setEnabled(False)  # 重置
        self.ui.pushButton_start.clicked.connect(self.nasStartBtnCallback)  # 确定
        self.ui.pushButton_reset.clicked.connect(self.resetStartBtnCallback)  # 重置
        self.ui.pushButton_hyper_param.clicked.connect(self.getHyperparametersSetting)  # 超参数设置
        self.ui.pushButton_refresh.clicked.connect(self.modelVisBtnCallback)  # 显示特征图
        # 在线与离线模式选择框
        self.ui.checkBox_online.clicked.connect(self.offlineClickedCallback)  # 离线
        self.ui.checkBox_offline.clicked.connect(self.onlineClickedCallback)  # 在线

        self.output_path = None
        self.new_method = False
        self.htmlLoadFinished = False  # 指示是否已经创建Echarts
        self.display_output_path = False
        self.ui.tabWidget.tabBarClicked.connect(self.tabWidgetClickedCallback)

        self.modelVis = ModelViewer(self.ui.scrollArea_feature_map, img_size=[32, 32])

        # move to center
        desktop = QApplication.desktop()
        ox, oy = (desktop.width() - self.width()) // 2, (desktop.height() - self.height()) // 2
        self.move(ox, oy)

        self.showMaximized()
        self.show()  # QMainWindow必须调用该函数，但QDialog不用

    def onlineClickedCallback(self):
        self.ui.checkBox_online.setChecked(not self.ui.checkBox_offline.isChecked())

    def offlineClickedCallback(self):
        self.ui.checkBox_offline.setChecked(not self.ui.checkBox_online.isChecked())

    def _classify_infer(self, first_init=False):
        # pass
        if first_init:
            genotype = eval("genotypes.%s_cifar10_2" % self.cur_method)
            self.model = Network(C=36, num_classes=10, layers=20, auxiliary=True, genotype=genotype)
            self.model.load_state_dict(
                torch.load('%s/%s/checkpoint/%s_cifar10_2_retrain_best.pt' %
                           (C.cache_dir, self.cur_method, self.cur_method), map_location='cpu'))
            self.model.drop_path_prob = 0.2
            self.model.eval()

            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
            test_data = dset.CIFAR10(root='%s/data/cifar10' % C.cache_dir, train=False, download=False,
                                     transform=test_transform)
            self.test_queue = DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=2, shuffle=False)
            self.test_iter = iter(self.test_queue)

            visual_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            visual_data = dset.CIFAR10(root='%s/data/cifar10' % C.cache_dir, train=False, download=False,
                                       transform=visual_transform)
            self.visual_queue = DataLoader(visual_data, batch_size=1, pin_memory=True, num_workers=2, shuffle=False)
            self.visual_iter = iter(self.visual_queue)

        else:
            input, target = next(self.test_iter)
            start_time = time.time()
            pred = self.model(input)
            end_time = time.time()
            target_class = target.item()
            pred_class = torch.argmax(pred[0]).item()

            # show
            self.ui.lineEdit.setText(self.cur_method)
            self.ui.lineEdit_2.setText(str(target_class) + '-' + CIFAR10_CLASS[target_class])
            self.ui.lineEdit_3.setText(str(pred_class) + '-' + CIFAR10_CLASS[pred_class])
            self.ui.lineEdit_4.setText('%.3f s' % (end_time - start_time))

            input, target = next(self.visual_iter)
            img = input.detach().squeeze().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = np.uint8(img * 255)

            ratio = self.ui.graphicsView.height() / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            self.infer_img_scene.clear()
            widthStep = width * 3
            infer_frame = QtGui.QImage(img.data, width, height, widthStep, QtGui.QImage.Format_RGB888)
            self.infer_img_scene.addPixmap(QtGui.QPixmap.fromImage(infer_frame))
            self.infer_img_scene.update()
            self.ui.graphicsView.setScene(self.infer_img_scene)
            # self.ui.verticalLayout.addWidget(self.ui.graphicsView)
    
    def _classify_infer_burgerformer(self, first_init=False):
        # pass
        if first_init:
            batch_size = 1
            # model
            self.model = create_model(
                'unifiedarch',
                num_classes=100,
                net_config=eval("arch.%s" % 'burgerformer_starlight'),
            )
            self.model.eval()
            resume_checkpoint(self.model, os.path.join(C.cache_dir, self.cur_method, 'checkpoint/net.pth'), optimizer=None, loss_scaler=None, log_info=False)
            torch.save(self.model, os.path.join(C.cache_dir, self.cur_method, 'checkpoint/model.pth'))

            # data loader
            dataset_eval = create_dataset(
                "", 
                root='%s/data/ImageNet-100' % C.cache_dir, 
                split="val", 
                is_training=False, 
                class_map="", 
                download=False, 
                batch_size=batch_size
            )
            data_loader_val = create_loader(
                dataset_eval,
                input_size=(3, 224, 224),
                batch_size=batch_size,
                is_training=False,
                use_prefetcher=True,
                interpolation='bicubic',
                mean=(0.485, 0.456, 0.406), # imagenet
                std=(0.229, 0.224, 0.225),
                num_workers=4,
                distributed=False,
                crop_pct=0.875,
                pin_memory=True,
                shuffle=True,
            )
            self.test_queue = data_loader_val
            self.test_iter = iter(self.test_queue)

            data_loader_val = create_loader(
                dataset_eval,
                input_size=(3, 224, 224),
                batch_size=batch_size,
                is_training=False,
                use_prefetcher=True,
                interpolation='bicubic',
                mean=(0, 0, 0), # imagenet
                std=(1, 1, 1),
                num_workers=4,
                distributed=False,
                crop_pct=0.875,
                pin_memory=True,
            )
            self.visual_queue = data_loader_val
            self.visual_iter = iter(self.visual_queue)

        else:
            input, target = next(self.test_iter)
            input, target = input.cpu(), target.cpu()
            start_time = time.time()
            pred = self.model(input)
            end_time = time.time()
            target_class = target.item()
            pred_class = torch.argmax(pred[0]).item()

            # show
            self.ui.lineEdit.setText(self.cur_method)
            self.ui.lineEdit_2.setText('class: {}'.format(target_class))
            self.ui.lineEdit_3.setText('class: {}'.format(pred_class))
            self.ui.lineEdit_4.setText('%.3f s' % (end_time - start_time))

            input, target = next(self.visual_iter)
            input, target = input.cpu(), target.cpu()
            img = input.detach().squeeze().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = np.uint8(img * 255)

            ratio = self.ui.graphicsView.height() / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            self.infer_img_scene.clear()
            widthStep = width * 3
            infer_frame = QtGui.QImage(img.data, width, height, widthStep, QtGui.QImage.Format_RGB888)
            self.infer_img_scene.addPixmap(QtGui.QPixmap.fromImage(infer_frame))
            self.infer_img_scene.update()
            self.ui.graphicsView.setScene(self.infer_img_scene)
            # self.ui.verticalLayout.addWidget(self.ui.graphicsView)

    def initConfig(self, first_init=False, online_change=False):
        # 数据集和方法
        self.ui.comboBox_dataset.setCurrentIndex(0)
        self.ui.comboBox_method.setCurrentIndex(0)
        # 在线和离线
        self.ui.checkBox_online.setChecked(False)
        self.ui.checkBox_offline.setChecked(False)

        # # 训练参数曲线
        # # learning_rate
        # self.lr_LineHtml = QWebEngineView()
        # self.ui.scrollArea_lr.setWidget(self.lr_LineHtml)
        # htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "lr_LineHtml.html"))
        # self.lr_LineHtml.load(QUrl(convertPath(htmlFilename)))
        # # train_loss
        # self.train_loss_LineHtml = QWebEngineView()
        # self.ui.scrollArea_train_loss.setWidget(self.train_loss_LineHtml)
        # htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "train_loss_LineHtml.html"))
        # self.train_loss_LineHtml.load(QUrl(convertPath(htmlFilename)))
        # # train_acc
        # self.train_acc_LineHtml = QWebEngineView()
        # self.ui.scrollArea_train_acc.setWidget(self.train_acc_LineHtml)
        # htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "train_acc_LineHtml.html"))
        # self.train_acc_LineHtml.load(QUrl(convertPath(htmlFilename)))
        # # valid_loss
        # self.valid_loss_LineHtml = QWebEngineView()
        # self.ui.scrollArea_valid_loss.setWidget(self.valid_loss_LineHtml)
        # htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "valid_loss_LineHtml.html"))
        # self.valid_loss_LineHtml.load(QUrl(convertPath(htmlFilename)))
        # # valid_acc
        # self.valid_acc_LineHtml = QWebEngineView()
        # self.ui.scrollArea_valid_acc.setWidget(self.valid_acc_LineHtml)
        # htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "valid_acc_LineHtml.html"))
        # self.valid_acc_LineHtml.load(QUrl(convertPath(htmlFilename)))

        if online_change:
            self.online_switch = True
            return
        self.start_online = 1

        # 重置全局计数器
        global counter
        counter = 0
    
    def initHtml(self):
        print(self.getData('dataset'))
        if self.getData('dataset') == 'null':
            pass
        elif self.getData('dataset') == 'CIFAR-10':
            # 训练参数曲线
            # learning_rate
            self.lr_LineHtml = QWebEngineView()
            self.ui.scrollArea_lr.setWidget(self.lr_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "lr_LineHtml.html"))
            self.lr_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # train_loss
            self.train_loss_LineHtml = QWebEngineView()
            self.ui.scrollArea_train_loss.setWidget(self.train_loss_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "train_loss_LineHtml.html"))
            self.train_loss_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # train_acc
            self.train_acc_LineHtml = QWebEngineView()
            self.ui.scrollArea_train_acc.setWidget(self.train_acc_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "train_acc_LineHtml.html"))
            self.train_acc_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # valid_loss
            self.valid_loss_LineHtml = QWebEngineView()
            self.ui.scrollArea_valid_loss.setWidget(self.valid_loss_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "valid_loss_LineHtml.html"))
            self.valid_loss_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # valid_acc
            self.valid_acc_LineHtml = QWebEngineView()
            self.ui.scrollArea_valid_acc.setWidget(self.valid_acc_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "valid_acc_LineHtml.html"))
            self.valid_acc_LineHtml.load(QUrl(convertPath(htmlFilename)))
        elif self.getData('dataset') == 'ImageNet-100':
            # BurgerFormer曲线
            # accuracy
            self.bf_acc_LineHtml = QWebEngineView()
            self.ui.scrollArea_train_loss.setWidget(self.bf_acc_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "burgerformer_acc_LineHtml.html"))
            self.bf_acc_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # flops
            self.bf_flops_LineHtml = QWebEngineView()
            self.ui.scrollArea_train_acc.setWidget(self.bf_flops_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "burgerformer_flops_LineHtml.html"))
            self.bf_flops_LineHtml.load(QUrl(convertPath(htmlFilename)))
            # parameters
            self.bf_params_LineHtml = QWebEngineView()
            self.ui.scrollArea_lr.setWidget(self.bf_params_LineHtml)
            htmlFilename = "file:///{}".format(os.path.join(C.html_dir, "burgerformer_params_LineHtml.html"))
            self.bf_params_LineHtml.load(QUrl(convertPath(htmlFilename)))

    def tabWidgetClickedCallback(self):
        self.resizeEvent(None)

    def getData(self, type):
        if type == 'dataset':
            return self.ui.comboBox_dataset.currentText()
        elif type == 'method':
            return self.ui.comboBox_method.currentText()
        elif type == 'is_online':
            return self.ui.checkBox_online.isChecked()
        elif type == 'is_offline':
            return self.ui.checkBox_offline.isChecked()

    def checkData(self):
        if self.getData('dataset') == 'null':  # 如果数据集和模型为null，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select a dataset')
            return False
        if self.getData('is_online') is False and self.getData('is_offline') is False:  # 如果在线离线模式为空，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select online or offline mode')
            return False
        if self.getData('method') == 'null':  # 如果剪枝为null且量化为null，弹出提醒框
            self.jumpQMessageBox("Message", 'Please select a NAS algorithm')
            return False

        return True

    def run_algo(self):
        """
        This function sends the running command to run the NAS algorithm online.

        Returns
        -------
        None
        """
        global sub_process

        if self.getData('method') == 'BurgerFormer':
            hyper_parameters = self.diag.get_dict_data()
            iteration = int(hyper_parameters['iteration'])
            target_flops = float(hyper_parameters['target_flops'])
            target_params = float(hyper_parameters['target_params'])

            debug = 1
            command = 'bash %s %s %s %s %s %s' % \
                    (os.path.join(C.current_dir, 'nas_burgerformer/online_run.sh'), C.work_dir, iteration, target_flops, target_params, debug)
            sub_process = subprocess.Popen(command, shell=True)
        else:
            # command = 'bash {}'.format('./darts_online_debug/debug_online.sh') + \
            #           ' {}'.format(self.args['dataset']) + \
            #           ' {}'.format(self.args['method'])
            debug = False
            hyper_parameters = self.diag.get_dict_data()
            epochs = int(hyper_parameters['epochs'])
            log_name = '{}/{}/logdir/Online_{}_{}.log'.format(
                C.cache_dir, self.cur_method, self.cur_method, dataset)
            if os.path.exists(log_name):
                os.remove(log_name)

            command = 'bash %s %s %s %s %s' % \
                    (os.path.join(C.current_dir, 'online/online_run.sh'), self.cur_method, C.work_dir, debug, epochs)
            sub_process = subprocess.Popen(command, shell=True)

            self.finish_timer.start(5000) # 5s

        # print('[PID: %s] %s' % (sub_process.pid, command))

    def check_online_finished(self):
        log_name = '{}/{}/logdir/Online_{}_{}.log'.format(
            C.cache_dir, self.cur_method, self.cur_method, dataset)
        with open(log_name) as f:
            lines = f.readlines()
        is_finished = False
        for line in lines: # 检查程序是否运行完成
            if "END OF ALL !!!" in line:
                is_finished = True
                break
        if is_finished:
            self.finish_timer.stop()
            QMessageBox.information(self, "Complete", f"{self.cur_method} online mode end !")
            if self.cur_method == 'DARTS':
                for line in lines[::-1]:
                    if "valid_acc" in line:
                        line_break = line.split()
                        valid_acc = line_break[line_break.index("valid_acc")+1]
                        valid_acc = float(valid_acc)
                        break
                for line in lines[::-1]:
                    if "param size = " in line:
                        line_break = line.split()
                        param_size = line_break[line_break.index("=")+1]
                        param_size = float(param_size)
                        break
                self.ui.label_results.setText(f'Params: {param_size:.1f}M  Top-1: {valid_acc:.1f}')
            elif self.cur_method == 'GDAS':
                for line in lines[::-1]:
                    if "valid_acc" in line:
                        line_break = line.split()
                        valid_acc = line_break[line_break.index("valid_acc")+1]
                        valid_acc = float(valid_acc)
                        break
                for line in lines[::-1]:
                    if "Params = " in line:
                        line_break = line.split()
                        param_size = line_break[line_break.index("Params")+2]
                        param_size = float(param_size)
                        break
                self.ui.label_results.setText(f'Params: {param_size:.1f}M  Top-1: {valid_acc:.1f}')
            elif self.cur_method == 'BurgerFormer':
                self.ui.label_results.setText('FLOPs: 1.1G\nParams: 10.1M\nTop-1: 87.16')
            else:
                raise NotImplementedError

    def get_online_pid(self, cmd_key):
        process_list = list(psutil.process_iter())
        regex = "pid=(\d+),\sname=\'" + "python" + "\'"
        ini_regex = re.compile(regex)
        for p in process_list:
            process_info = str(p)
            result = ini_regex.search(process_info)
            if result is not None:
                argv = p.cmdline()
                for arg in argv:
                    if cmd_key in arg:
                        return p.pid
        return None

    def kill_online(self, pid):
        choice = QMessageBox.No
        p = psutil.Process(pid)
        choice = QMessageBox.question(self, "Warning", f'Do you want to stop the online training? [PID: {pid}]',
                                      QMessageBox.Yes | QMessageBox.No)  # 1
        if choice == QMessageBox.Yes:
            p.kill()
            return True  # prcoess killed
        return False  #

    # 搜索确认按钮回调函数
    def nasStartBtnCallback(self):

        # 逻辑检查
        if not self.checkData():
            return
        
        if self.getData('method') == 'BurgerFormer': # 隐藏
            self.ui.tabWidget.setTabEnabled(1, False)
        else:
            self.ui.tabWidget.setTabEnabled(1, True)

        self.initHtml()

        # 调整按钮及左边窗显示
        self.display = True
        self.cur_method = self.getData('method')
        global update_threshold
        if self.cur_method == 'DARTS':
            update_threshold = 50
        elif self.cur_method == 'GDAS':
            update_threshold = 250
        elif self.cur_method == 'BurgerFormer':
            update_threshold = 20
        else:
            raise NotImplementedError

        # 设置超参数
        if self.ui.textBrowser.toPlainText() == '':
            self.getHyperparametersSetting()
        
        # 在线和离线模式区分
        if self.getData('is_online'):
            self.online_data_dict = None
            log_name = '{}/{}/logdir/Online_{}_{}.log'.format(
                C.cache_dir, self.cur_method, self.cur_method, dataset)
            if os.path.exists(log_name):
                choice = QMessageBox.question(self, "Message", 'Files exist in this path, do you want to delete them?', QMessageBox.Yes | QMessageBox.No)  # 1
                if choice == QMessageBox.Yes:
                    os.remove(log_name)
                else:
                    return
            self.ui.label_status.setText('%s is running online' % self.cur_method)
            self.ui.label_results.setText('A fine-tuning process is needed after the online search')
            self.run_algo()
        else:
            self.ui.label_status.setText('%s is running offline' % self.cur_method)
            if self.cur_method == 'DARTS':
                self.ui.label_results.setText('Params: 2.5M\tTop-1: 97.01')
            elif self.cur_method == 'GDAS':
                self.ui.label_results.setText('Params: 3.9M\tTop-1: 97.30')
            elif self.cur_method == 'BurgerFormer':
                self.ui.label_results.setText('FLOPs: 1.1G\nParams: 10.1M\nTop-1: 87.16')
            else:
                raise NotImplementedError

        # 模型曲线展示
        if self.getData('dataset') == 'CIFAR-10':
            self.buildChart(self.lr_LineHtml, self.ui.scrollArea_lr)
            self.buildChart(self.train_loss_LineHtml, self.ui.scrollArea_train_loss)
            self.buildChart(self.train_acc_LineHtml, self.ui.scrollArea_train_acc)
            self.buildChart(self.valid_loss_LineHtml, self.ui.scrollArea_valid_loss)
            self.buildChart(self.valid_acc_LineHtml, self.ui.scrollArea_valid_acc)
        elif self.getData('dataset') == 'ImageNet-100':
            self.buildChart(self.bf_params_LineHtml, self.ui.scrollArea_lr)
            self.buildChart(self.bf_acc_LineHtml, self.ui.scrollArea_train_loss)
            self.buildChart(self.bf_flops_LineHtml, self.ui.scrollArea_train_acc)
        self.htmlLoadFinished = True
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_reset.setEnabled(True)

        # 开启数据处理线程
        args = copy.deepcopy(self.diag.get_dict_data())
        args['dataset'] = self.getData('dataset')
        args['method'] = self.getData('method')
        args['is_online'] = self.getData('is_online')
        args['is_offline'] = self.getData('is_offline')
        self.nasVisTask = NASVisTask(args)
        self.nasVisTask.signalUpdateUi.connect(self.updateUiCallback)  # 关联数据处理线程和可视化线程
        self.nasVisTask.start()

        # 在线推理
        if self.cur_method == 'BurgerFormer':
            self._classify_infer_burgerformer(first_init=True)
        else:
            self._classify_infer(first_init=True)

    # 搜索重置按钮回调函数
    def resetStartBtnCallback(self):
        # 如果程序运行，弹出提醒框，Message用户手动关闭程序，返回
        pid = None
        if self.getData('is_online'):
            pid = self.get_online_pid(self.cur_method)
            if pid is not None and not self.kill_online(pid):
                return

        # 把数据集、模型、在线/离线模式和压缩方法变成缺省值，清空在线离线模式，超参数字图，清空柱状图
        self.display = False
        self.ui.label_status.setText('')  # 状态显示
        self.ui.label_results.setText('')  # 结果输入
        self.ui.textBrowser.setText('')  # 超参数
        self.resetHtml()
        self.initHtml()
        self.initConfig()

        # “重置”按钮变灰，此时“确认”按钮应该是从灰色变成激活状态
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_reset.setEnabled(False)

        # # 关闭数据处理线程
        # self.nasVisTask.terminate()

    def resetHtml(self):
        if self.getData('dataset') == 'CIFAR-10':
            self.train_loss_LineHtml.destroy()
            self.train_acc_LineHtml.destroy()
            self.valid_loss_LineHtml.destroy()
            self.valid_acc_LineHtml.destroy()
        elif self.getData('dataset') == 'ImageNet-100':
            self.bf_params_LineHtml.destroy()
            self.bf_acc_LineHtml.destroy()
            self.bf_flops_LineHtml.destroy()


    def jumpQMessageBox(self, title, message):
        box = QMessageBox(QMessageBox.Warning, title, message)
        box.addButton("OK", QMessageBox.YesRole)
        box.exec_()

    def stopBtnCallback(self):
        os.system('kill -9 `ps -ef |grep darts_online|awk \'{print $2}\' `')
        self.online_killed = True

    def modelVisBtnCallback(self):
        model_path = os.path.join(C.cache_dir, self.cur_method, 'checkpoint/model.pth')
        model = torch.load(model_path, map_location=self.device)
        model.drop_path_prob = 0.0
        model.eval()
        if self.cur_method == 'BurgerFormer':
            self.modelVis.img_size = [224, 224]
        else:
            self.modelVis.img_size = [32, 32]
        self.modelVis.slotUpdateModel(model, os.path.join(C.quiver_data, 'CIFAR-10'))

    # 超参设置按钮回调函数
    def getHyperparametersSetting(self):
        if self.getData('method') == 'null':
            self.jumpQMessageBox("Message", 'Please select a NAS algorithm')
            return
        self.diag = HyperparametersSettingWindow(self.cur_method)
        output_dir = os.path.join(C.cache_dir, '%s' % (self.getData('method')))
        self.diag.ui.lineEdit_5.setText(output_dir)
        if self.cur_method == 'GDAS':
            self.diag.ui.lineEdit_4.setText("250")
        if self.diag.exec_():
            data = self.diag.get_data()
            datastr = ''
            for d in data:
                datastr += d + '\n'
            self.ui.textBrowser.setText(datastr)
        self.output_dir = self.diag.ui.lineEdit_5.text()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def buildChart(self, htmlViewer, container):
        htmlViewer.page().runJavaScript(
            "buildChart('{}', '{}'); ".format(int(container.width()), int(container.height())))

    def resizeEvent(self, a0):

        num = self.ui.tabWidget.count()
        cur_indx = self.ui.tabWidget.currentIndex()
        for i in range(num):
            self.ui.tabWidget.setCurrentIndex(i)
        self.ui.tabWidget.setCurrentIndex(cur_indx)

        if self.htmlLoadFinished:
            # Method Criterion
            if self.getData('dataset') == 'CIFAR-10':
                self.buildChart(self.lr_LineHtml, self.ui.scrollArea_lr)
                self.buildChart(self.train_loss_LineHtml, self.ui.scrollArea_train_loss)
                self.buildChart(self.train_acc_LineHtml, self.ui.scrollArea_train_acc)
                self.buildChart(self.valid_loss_LineHtml, self.ui.scrollArea_valid_loss)
                self.buildChart(self.valid_acc_LineHtml, self.ui.scrollArea_valid_acc)
            elif self.getData('dataset') == 'ImageNet-100':
                self.buildChart(self.bf_params_LineHtml, self.ui.scrollArea_lr)
                self.buildChart(self.bf_acc_LineHtml, self.ui.scrollArea_train_loss)
                self.buildChart(self.bf_flops_LineHtml, self.ui.scrollArea_train_acc)

        return super().resizeEvent(a0)

    def _showImage(self, img_path, graphicsView, show_type):
        img = cv2.imread(str(img_path))
        graph_h, graph_w = graphicsView.height(), graphicsView.width()
        ratio_h = graph_h / img.shape[0]
        ratio_w = graph_w / img.shape[1]
        if ratio_h < ratio_w:
            ratio = ratio_h
        else:
            ratio = ratio_w
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        if show_type == 'normal':
            scene = self.normal_scene
        elif show_type == 'reduce':
            scene = self.reduce_scene
        else:
            raise ValueError('No Defined Scene!')
        scene.clear()
        widthStep = img_w * 3
        frame = QtGui.QImage(img.data, img_w, img_h, widthStep, QtGui.QImage.Format_RGB888)
        item = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(frame))
        item.setOffset(-img_w / 2.0, 0)
        scene.addItem(item)
        graphicsView.setScene(scene)
        graphicsView.show()

    def _online_check_datadict_update(self, cur):
        for _key in method_index_list:
            if _key == 'genotype':
                continue
            if len(cur[_key]) != len(self.online_data_dict[_key]):
                return True
        return False

    def _show_latest_data(self, cur):
        # print(cur)
        for _key in method_index_list:
            if len(cur[_key]) != len(self.online_data_dict[_key]):
                if _key == 'genotype':
                    if (len(cur[_key]) - 1) < update_threshold:
                        # Genotype Visualization
                        self._showImage('{}/{}/figure/{}/{}_normal_cell.png'.format(
                            C.cache_dir, self.cur_method, self.cur_method, len(cur[_key]) - 1),
                            self.ui.graphicsView_2, show_type='normal'
                        )
                        self._showImage(
                            '{}/{}/figure/{}/{}_reduction_cell.png'.format(
                                C.cache_dir, self.cur_method, self.cur_method, len(cur[_key]) - 1),
                            self.ui.graphicsView_3, show_type='reduce'
                        )
                elif _key == 'epoch':
                    continue
                else:
                    # print(_key, cur[_key])
                    eval('self.%s_LineHtml' % _key).page().runJavaScript('''
                            index_data.push({})
                            loss_data.push({})
                        '''.format(len(cur[_key]), cur[_key][-1]))

        self.online_data_dict = cur

    def _first_show_online_data(self, cur):
        for _key in method_index_list:
            if len(cur[_key]) != 0:
                for i in range(len(cur[_key])):
                    if _key == 'genotype':
                        # Genotype Visualization
                        self._showImage(
                            '{}/{}/figure/{}/{}_normal_cell.png'.format(C.cache_dir, self.cur_method, self.cur_method,
                                                                        i),
                            self.ui.graphicsView_2, show_type='normal')
                        self._showImage(
                            '{}/{}/figure/{}/{}_reduction_cell.png'.format(C.cache_dir, self.cur_method,
                                                                           self.cur_method, i),
                            self.ui.graphicsView_3, show_type='reduce')
                    elif _key == 'epoch':
                        continue
                    else:
                        # print(_key, cur[_key])
                        eval('self.%s_LineHtml' % _key).page().runJavaScript('''
                                index_data.push({})
                                loss_data.push({})
                            '''.format(i, cur[_key][i]))

    def updateUiCallback(self):
        global counter
        global update_threshold

        xValue = counter

        if self.display:
            if self.getData('is_offline'):
                if self.cur_method in ['BurgerFormer']:
                    global acc_iter
                    global flops_iter
                    global params_iter
                    self.bf_acc_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, acc_iter))
                    self.bf_flops_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, flops_iter))
                    self.bf_params_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, params_iter))
                    self._classify_infer_burgerformer(first_init=False)
                else:
                    # Method Criterion
                    self.lr_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, lr_epoch))
                    self.train_loss_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, train_loss_epoch))
                    self.train_acc_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, train_acc_epoch))
                    self.valid_loss_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, valid_loss_epoch))
                    self.valid_acc_LineHtml.page().runJavaScript(
                        '''index_data.push({})\nloss_data.push({})'''.format(xValue, valid_acc_epoch))
                    self._classify_infer(first_init=False)
                    # Genotype Visualization
                    self._showImage('{}/{}/figure/{}/{}_normal_cell.png'.format(
                        C.cache_dir, self.cur_method, self.cur_method,
                        counter if counter < update_threshold else update_threshold - 1),
                        self.ui.graphicsView_2, show_type='normal')
                    self._showImage('{}/{}/figure/{}/{}_reduction_cell.png'.format(
                        C.cache_dir, self.cur_method, self.cur_method,
                        counter if counter < update_threshold else update_threshold - 1),
                        self.ui.graphicsView_3, show_type='reduce')
            elif self.getData('is_online'):
                if self.cur_method in ['BurgerFormer']:
                    cur_data_dict = from_burgerformer_read_log(method=self.cur_method,
                                                        log_name='{}/{}/logdir/Online_{}_{}.log'
                                                        .format(C.cache_dir, 'BurgerFormer', 'BurgerFormer', 'ImageNet-100'),
                                                        key_words=['acc', 'flops', 'params'])
                    data_len = len(cur_data_dict['acc'])
                    if counter < data_len:
                        counter += 1

                        xValue = counter
                        acc_iter = cur_data_dict['acc'][counter - 1]
                        flops_iter = cur_data_dict['flops'][counter - 1]
                        params_iter = cur_data_dict['flops'][counter - 1]

                        self.bf_acc_LineHtml.page().runJavaScript(
                            '''index_data.push({})\nloss_data.push({})'''.format(xValue, acc_iter))
                        self.bf_flops_LineHtml.page().runJavaScript(
                            '''index_data.push({})\nloss_data.push({})'''.format(xValue, flops_iter))
                        self.bf_params_LineHtml.page().runJavaScript(
                            '''index_data.push({})\nloss_data.push({})'''.format(xValue, params_iter))
                else:
                    cur_data_dict = from_darts_read_log(method=self.cur_method,
                                                        log_name='{}/{}/logdir/Online_{}_{}.log'
                                                        .format(C.cache_dir, self.cur_method, self.cur_method, dataset),
                                                        key_words=method_index_list)
                    if self.online_data_dict is None:
                        self._first_show_online_data(cur_data_dict)
                        self.online_data_dict = cur_data_dict
                    elif self._online_check_datadict_update(cur_data_dict):
                        self._show_latest_data(cur_data_dict)

    def closeEvent(self, event):  # 函数名固定不可变
        pid = None
        if self.getData('is_online'):
            pid = self.get_online_pid(self.cur_method)
            if pid is not None and not self.kill_online(pid):
                event.ignore()  # 保留窗口
                return
        event.accept()  # 关闭窗口


if __name__ == "__main__":
    # Check img exist
    for _method in method_list:
        if _method == 'BurgerFormer':
            pass
        else:
            if not glob.glob('{}/{}/figure/{}/*.png'.format(C.cache_dir, _method, _method)):
                from nas_vis.model_vis import generate_all_genotypes

                generate_all_genotypes(C.cache_dir, method_list, method_index_list, dataset='cifar10', seed=2)

    app = QApplication(sys.argv)
    mainW = MainWindow()
    sys.exit(app.exec_())
