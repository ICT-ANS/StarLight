from PyQt5.QtWidgets import  QScrollArea, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import  QUrl, QTimer
import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine import server
from engine.model_utils import register_hook

import threading, time

class ModelViewer():
    def __init__(self, qtcontainer, img_size=None) -> None:
        """The easy interface for quiver 

        Parameters
        ----------
        qtcontainer : _type_
            recommand QScroolAera
        img_size : list, optional
            specify [height, width] of image for DNN, by default None
        """        
        self.myHtml = QWebEngineView()
        self.myHtml.loadFinished.connect(self.slotHtmlLoadFinished)

        self.container = qtcontainer
        self.container.setWidget(self.myHtml)

        self.thread = None # http server 运行线程

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.slotTimeout)
        self.timer.start()

        if img_size is None:
            self.img_size = [200, 200]
        else:
            self.img_size = img_size

        self.http_server = None
        pass
    
    def set_img_size(self, img_size:list):
        """set image size

        Parameters
        ----------
        img_size : list
            [height, width]
        """        
        self.img_size = img_size

    def slotTimeout(self):
        """wait for http_server started
        """        
        if self.http_server is not None:
            if self.http_server.started: # 等待 http_server 完成启动
                self.htmlLoadFinished = False
                # print("server started")
                self.myHtml.load(QUrl("http://localhost:5000/"))
                
    def slotHtmlLoadFinished(self):
        """slot for html load finished
        """        
        self.htmlLoadFinished = True
        self.timer.stop() #stop executing slotYimeout
        # print("timer stop")
    
    def slotUpdateModel(self, model, datapath:str):
        """load & update model for visulization

        Parameters
        ----------
        model : Any
            pytorch model
        datapath : str
            image path for visulization
        """        
        hook_list = register_hook(model)

        self.timer.start()

        if self.thread is None:
            server._http_server = None
            self.thread = threading.Thread(target=server.launch, args=(model, hook_list, datapath, False, self.img_size, ))
            self.thread.daemon = True
            self.thread.start()

            while server._http_server is None:
                time.sleep(0.2)

            self.http_server = server._http_server

        server.update_model(model, hook_list, datapath, self.img_size)
