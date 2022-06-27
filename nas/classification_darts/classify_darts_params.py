import os
import sys
from collections import OrderedDict

from PyQt5.QtWidgets import (QMessageBox, QDialog)

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
from ui.qtui.nas.darts_params_setting_ui import *


# 超参输入界面
class HyperparametersSettingWindow(QDialog):
    def __init__(self):
        """
        This class defines UI and other basic parameters.
        """
        QDialog.__init__(self)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.reset()
        self.ui.pushButton.clicked.connect(self.select_path)
        self.ui.pushButton_2.clicked.connect(self.ok)
        self.ui.pushButton_3.clicked.connect(self.reset)
        self.ui.pushButton_4.clicked.connect(self.quit)

    def select_path(self):
        """
        Select folder.

        Returns
        -------
        str
            folder name
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        self.ui.lineEdit_5.setText(path)

    def ok(self):
        """
        This function checks if the info is valid

        Returns
        -------
        None
        """
        valid, info = self.valid_check()
        if valid:
            self.accept()
        else:
            msg_box = QMessageBox(QMessageBox.Warning, "Alert", info)
            msg_box.exec_()

    def reset(self):
        """
        This function reset the line edit.

        Returns
        -------
        None
        """
        self.ui.lineEdit_6.setText("10")
        self.ui.lineEdit_4.setText("50")
        self.ui.lineEdit_2.setText("64")
        self.ui.lineEdit_7.setText("0.025")
        self.ui.lineEdit.setText("3e-4")
        self.ui.lineEdit_3.setText("2")

    def quit(self):
        """
        This function quit the UI.

        Returns
        -------
        None
        """
        self.reject()

    def get_data(self):
        """
        Get the corresponding data.

        Returns
        -------
        list
            data content
        """
        data = []
        data.append('classes：' + self.ui.lineEdit_6.text())
        data.append('epochs：' + self.ui.lineEdit_4.text())
        data.append('batch size：' + self.ui.lineEdit_2.text())
        data.append('learning rate：' + self.ui.lineEdit_7.text())
        data.append('arch learning rate：' + self.ui.lineEdit.text())
        data.append('seed：' + self.ui.lineEdit_3.text())
        data.append('output dir：' + self.ui.lineEdit_5.text())
        return data

    def get_dict_data(self):
        """
        Converts data to dictionary form.

        Returns
        -------
        dict
            hyper-parameters in the dictionary form
        """
        data = {}
        data['classes'] = self.ui.lineEdit_6.text()
        data['epochs'] = self.ui.lineEdit_4.text()
        data['bs：'] = self.ui.lineEdit_2.text()
        data['lr：'] = self.ui.lineEdit_7.text()
        data['arch_lr'] = self.ui.lineEdit.text()
        data['seed'] = self.ui.lineEdit_3.text()
        data['output_dir'] = self.ui.lineEdit_5.text()
        return data

    def valid_check(self):
        """
        Check data format

        Returns
        -------
        str
            True，ok or other questions
        """
        def is_float(data):
            """
            Judge whether it is float data.

            Parameters
            ----------
            data : float
                input data to be checked

            Returns
            -------
            bool
                true or false
            """
            try:
                float(data)
            except ValueError:
                return False
            return True

        def is_int(data):
            """
            Judge whether it is int data.

            Parameters
            ----------
            data : int
                input data to be checked

            Returns
            -------
            bool
                true or false
            """
            try:
                int(data)
            except ValueError:
                return False
            return True

        def is_in(data, minn=None, maxx=None, min_bound=True, max_bound=True):
            """
            Judge whether the data are included in each other.

            Parameters
            ----------
            data : float or int
                data to be checked
            minn : float or int
                minimum threshold
            maxx : float or int
                maximum threshold
            min_bound : bool
                true or false
            max_bound : bool
                true or false

            Returns
            -------
            bool
                true or false
            """
            if minn is not None:
                if data < minn or (not min_bound and data == minn):
                    return False
            if maxx is not None:
                if data > maxx or (not max_bound and data == maxx):
                    return False
            return True

        def is_path(path):
            return os.path.exists(path)

        data = OrderedDict()
        data['classes'] = self.ui.lineEdit_6.text()
        data['epochs'] = self.ui.lineEdit_4.text()
        data['batch size'] = self.ui.lineEdit_2.text()
        data['learning rate'] = self.ui.lineEdit_7.text()
        data['arch learning rate'] = self.ui.lineEdit.text()
        data['image size'] = self.ui.lineEdit_3.text()
        data['output path'] = self.ui.lineEdit_5.text()
        # validity test
        for key in data:
            if key in ['classes', 'epochs', 'batch size']:
                if not is_int(data[key]):
                    return False, '{}应输入整数'.format(key)
                if not is_in(int(data[key]), 0., min_bound=False):
                    return False, '{}应大于0'.format(key)
            elif key in ['learning rate', 'arch learning rate']:
                if not is_float(data[key]):
                    return False, '{}应输入实数'.format(key)
                if not is_in(float(data[key]), 0., min_bound=False):
                    return False, '{}应大于0'.format(key)
            elif key in ['output path']:
                if not is_path(data[key]):
                    return False, '{}路径不存在'.format(key)
        return True, 'ok'
