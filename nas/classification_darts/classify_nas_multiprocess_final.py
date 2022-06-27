from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem, QMessageBox)

import os
import copy
import cv2
import sys
import time
import glob
import psutil
import numpy as np
import subprocess
import torch
import torch.utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import threading

from config import C
from ui.qtui.nas.classify_nas_ui import *
from nas.classification_darts import classify_genotypes
from nas.classification_darts.classify_model_vis import modelVisTask
from nas.classification_darts.classify_model import NetworkCIFAR as Network
from nas.classification_darts.classify_darts_params import HyperparametersSettingWindow
from nas.classification_darts.classify_utils import *

# 用于数据交互的全局变量
counter = 0
uiData = 0
update_threshold = 50
sub_process = None
dataset = 'CIFAR-10'
CIFAR10_CLASS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
method_list = ['DARTS', 'GDAS']
method_index_list = ['lr', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'genotype', 'epoch']

data_dict = {}
for _m in method_list:
    data_dict[_m] = from_darts_read_log(method=_m,
                                        log_name='{}/{}/logdir/Offline_{}_{}.log'.format(C.cache_dir, _m, _m, dataset),
                                        key_words=method_index_list)

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
        thread = threading.Thread(target=self.run_algo)
        thread.start()

        while True:
            self.sleep(1)
            self.signalUpdateUi.emit()

    def run_algo(self):
        """
        This function sends the running command to run the algorithm.

        Returns
        -------
        None
        """
        global sub_process
        command = 'bash {}'.format('./darts_online_debug/debug_online.sh') + \
                  ' {}'.format(self.args['dataset']) + \
                  ' {}'.format(self.args['method'])
        sub_process = subprocess.Popen(command, shell=True)
        print('[PID: %s] %s' % (sub_process.pid, command))

    def offline(self):
        """
        This function runs the algorithm online.

        Returns
        -------
        None
        """
        offline_data = data_dict[self.args['method']]
        while True:
            global counter
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
                counter = update_threshold
            self.signalUpdateUi.emit()


# 主界面
class MainWindow(QMainWindow):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(MainWindow, self).__init__(parent=parent, flags=flags)
        self.normal_scene = QGraphicsScene()
        self.reduce_scene = QGraphicsScene()
        self.infer_img_scene = QGraphicsScene()
        self.online = False
        self.online_data_dict = None
        self.start_offline = 0
        self.online_running = False
        self.online_killed = False
        self.online_switch = False

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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

        # self.ui.tableWidget
        self.modelVisTask = None

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
            genotype = eval("classify_genotypes.%s_cifar10_2" % self.cur_method)
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

    def initConfig(self, first_init=False, online_change=False):
        # 数据集和方法
        self.ui.comboBox_dataset.setCurrentIndex(0)
        self.ui.comboBox_method.setCurrentIndex(0)
        # 在线和离线
        self.ui.checkBox_online.setChecked(False)
        self.ui.checkBox_offline.setChecked(False)

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

        if online_change:
            self.online_switch = True
            return
        self.start_online = 1

        # 重置全局计数器
        global counter
        counter = 0

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
            self.jumpQMessageBox("提示", '请选择数据集')
            return False
        if self.getData('is_online') is False and self.getData('is_offline') is False:  # 如果在线离线模式为空，弹出提醒框
            self.jumpQMessageBox("提示", '请选择在线或离线模式')
            return False
        if self.getData('method') == 'null':  # 如果剪枝为null且量化为null，弹出提醒框
            self.jumpQMessageBox("提示", '请选择搜索方法')
            return False

        return True

    # 搜索确认按钮回调函数
    def nasStartBtnCallback(self):

        # 逻辑检查
        if not self.checkData():
            return

        # 调整按钮及左边窗显示
        self.display = True
        self.cur_method = self.getData('method')
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_reset.setEnabled(True)
        if self.getData('is_online'):
            self.ui.label_status.setText('%s 方法在线展示中' % self.cur_method)
            self.ui.label_results.setText('在线搜索完成后需重训练搜得模型')
        else:
            self.ui.label_status.setText('%s 方法离线展示中' % self.cur_method)
            if self.cur_method == 'DARTS':
                self.ui.label_results.setText('Params: 2.5M\tTop-1: 97.01')
            elif self.cur_method == 'GDAS':
                self.ui.label_results.setText('Params: 3.9M\tTop-1: 97.30')
            else:
                raise NotImplementedError

        if self.ui.textBrowser.toPlainText() == '':
            self.getHyperparametersSetting()

        # 模型曲线展示
        self.buildChart(self.lr_LineHtml, self.ui.scrollArea_lr)
        self.buildChart(self.train_loss_LineHtml, self.ui.scrollArea_train_loss)
        self.buildChart(self.train_acc_LineHtml, self.ui.scrollArea_train_acc)
        self.buildChart(self.valid_loss_LineHtml, self.ui.scrollArea_valid_loss)
        self.buildChart(self.valid_acc_LineHtml, self.ui.scrollArea_valid_acc)
        self.htmlLoadFinished = True

        # 开启数据处理线程
        args = copy.deepcopy(self.diag.get_dict_data())
        args['dataset'] = self.getData('dataset')
        args['method'] = self.getData('method')
        args['is_online'] = self.getData('is_online')
        args['is_offline'] = self.getData('is_offline')
        self.nasVisTask = NASVisTask(args)
        self.nasVisTask.signalUpdateUi.connect(self.updateUiCallback)  # 关联数据处理线程和可视化线程
        self.nasVisTask.start()
        # infer
        self._classify_infer(first_init=True)

    # 搜索重置按钮回调函数
    def resetStartBtnCallback(self):
        # 如果程序运行，弹出提醒框，提示用户手动关闭程序，返回
        global sub_process
        if sub_process and psutil.pid_exists(sub_process.pid):
            self.jumpQMessageBox('在线程序正在运行', '请手动结束在线程序, PID: %s' % sub_process.pid)
            return

        # 把数据集、模型、在线/离线模式和压缩方法变成缺省值，清空在线离线模式，超参数字图，清空柱状图
        self.display = False
        self.ui.label_status.setText('')  # 状态显示
        self.ui.label_results.setText('')  # 结果输入
        self.ui.textBrowser.setText('')  # 超参数
        self.resetHtml()
        self.initConfig()

        # “重置”按钮变灰，此时“确认”按钮应该是从灰色变成激活状态
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_reset.setEnabled(False)

        # 关闭数据处理线程
        self.nasVisTask.terminate()

    def resetHtml(self):
        self.train_loss_LineHtml.destroy()
        self.train_acc_LineHtml.destroy()
        self.valid_loss_LineHtml.destroy()
        self.valid_acc_LineHtml.destroy()

    def jumpQMessageBox(self, title, message):
        box = QMessageBox(QMessageBox.Warning, title, message)
        box.addButton("确定", QMessageBox.YesRole)
        box.exec_()

    def stopBtnCallback(self):
        os.system('kill -9 `ps -ef |grep darts_online|awk \'{print $2}\' `')
        self.online_killed = True

    # def refreshBtnCallback(self):
    #     self.feature_map.load(QUrl(convertPath("http://localhost:5000")))
    def modelVisBtnCallback(self):
        if self.modelVisTask is not None:
            self.modelVisTask.stop()
            del self.modelVisTask
            self.modelVisHtml.close()
        self.modelVisHtml = QWebEngineView()
        self.ui.scrollArea_feature_map.setWidget(self.modelVisHtml)
        self.modelVishtmlFilename = "http://localhost:5050"
        cur_dataset = self.ui.comboBox_dataset.currentText()
        self.modelVisTask = modelVisTask(self, cur_dataset, 'DARTS_V1')
        self.modelVisTask.start()

    def modelVisLoad(self):
        # self.modelVisHtml.load(QUrl(convertPath(self.modelVishtmlFilename)))
        self.modelVisHtml.load(QUrl(self.modelVishtmlFilename))
        self.resizeEvent(None)

    # 超参设置按钮回调函数
    def getHyperparametersSetting(self):
        self.diag = HyperparametersSettingWindow()
        output_dir = os.path.join(C.cache_dir, '%s' % (self.getData('method')))
        self.diag.ui.lineEdit_5.setText(output_dir)
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
            self.buildChart(self.lr_LineHtml, self.ui.scrollArea_lr)
            self.buildChart(self.train_loss_LineHtml, self.ui.scrollArea_train_loss)
            self.buildChart(self.train_acc_LineHtml, self.ui.scrollArea_train_acc)
            self.buildChart(self.valid_loss_LineHtml, self.ui.scrollArea_valid_loss)
            self.buildChart(self.valid_acc_LineHtml, self.ui.scrollArea_valid_acc)

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
        if self.online_running and not self.online_killed:
            QtWidgets.QMessageBox.about(self, 'Online process running!', '请手动结束在线程序')
            event.ignore()
        else:
            event.accept()  # 关闭窗口


if __name__ == "__main__":
    # Check img exist
    for _method in method_list:
        if not glob.glob('{}/{}/figure/{}/*.png'.format(C.cache_dir, _method, _method)):
            from nas.classification_darts.classify_model_vis import generate_all_genotypes

            generate_all_genotypes(C.cache_dir, method_list, method_index_list, dataset='cifar10', seed=2)
    app = QApplication(sys.argv)

    mainW = MainWindow()

    sys.exit(app.exec_())
