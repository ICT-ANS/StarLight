from PyQt5.QtWidgets import  QScrollArea
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import  QUrl, QTimer
import numpy as np
offset = 20

default_colors = [
                '#dd6b66',
                '#759aa0',
                '#e69d87',
                '#8dc1a9',
                '#ea7e53',
                '#eedd78',
                '#73a373',
                '#73b9bc',
                '#7289ab',
                '#91ca8c',
                '#f49f42']

default_symbols = [
                'emptyCircle',
                'circle', 
                'rect', 
                'roundRect', 
                'triangle', 
                'diamond', 
                'pin', 
                'arrow', 
                'none']


class Line(object):
    '''
    @param: basehtml: absolute path of html file
    @param: qtcontainer: container to visualize html, only test on QScrollArea now
    @param: legends: list of str for legends
    '''
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, \
                 legends: list, colors=[], symbols=[], title="") -> None:
        super().__init__()

        self.myHtml = QWebEngineView()
        self.myHtml.loadFinished.connect(self.slotHtmlLoadFinished)

        self.container = qtcontainer
        self.container.setWidget(self.myHtml)

        self.htmlLoadFinished = False
    
        htmlFilename = "file:///{}".format(basehtml)
        self.myHtml.load(QUrl(htmlFilename))

        self.legends = legends
        self.title = title

        if len(colors) < len(legends):
            for i in range(len(colors), len(legends)):
                colors.append(np.random.choice(default_colors, 1)[0])
        self.colors = colors

        if len(symbols) < len(legends):
            for i in range(len(colors), len(legends)):
                symbols.append(np.random.choice(default_symbols, 1)[0])
        self.symbols = symbols

        self.resizeTimer = QTimer()
        self.resizeTimer.setInterval(300)
        self.resizeTimer.timeout.connect(self.slotResize)
        self.resizeTimer.start()

        self.width = self.container.width()
        self.height = self.container.height()

    def slotHtmlLoadFinished(self):
        self.htmlLoadFinished = True
        self.setTitle(self.title)
        self.build()
    
    def slotResize(self):
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()

    def update(self, xAxisData: any, yAxisData: list):
        assert len(self.legends) == len(yAxisData)
        yAxisData = [str(i) for i in yAxisData]
        xAxisData = str(xAxisData)
        jscode = '''update({}, {});'''.format(xAxisData, yAxisData)
        self.myHtml.page().runJavaScript(jscode)   

    def setTitle(self, title: str):
        jscode = '''setTitle('{}');'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        self.myHtml.page().runJavaScript("build('{}', '{}', {}, {}, {}); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset), 
                                                                    self.legends, self.colors, self.symbols))
    def clearData(self):
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        self.myHtml.page().runJavaScript("destroy();")
        pass      

class Bar(object):
    '''
    @param: basehtml: absolute path of html file
    @param: qtcontainer: container to visualize html, only test on QScrollArea now
    @param: legends: list of str for legends
    '''
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, \
                 xAxis: list, legends: list, colors=[], title="") -> None:
        super().__init__()

        self.myHtml = QWebEngineView()
        self.myHtml.loadFinished.connect(self.slotHtmlLoadFinished)

        self.container = qtcontainer
        self.container.setWidget(self.myHtml)

        self.htmlLoadFinished = False
    
        htmlFilename = "file:///{}".format(basehtml)
        self.myHtml.load(QUrl(htmlFilename))

        self.legends = legends
        self.series_num = len(self.legends)
        self.xAxis = xAxis
        self.title = title

        if len(colors) < len(legends):
            for i in range(len(colors), len(legends)):
                colors.append(np.random.choice(default_colors, 1)[0])
        self.colors = colors

        self.resizeTimer = QTimer()
        self.resizeTimer.setInterval(300)
        self.resizeTimer.timeout.connect(self.slotResize)
        self.resizeTimer.start()

        self.width = self.container.width()
        self.height = self.container.height()


    def slotHtmlLoadFinished(self):
        self.htmlLoadFinished = True
        self.setTitle(self.title)
        self.build()

    def slotResize(self):
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()
    
    def update(self, yAxisData: list):
        yAxisData = [str(i) for i in yAxisData]
        jscode = '''update({})'''.format(yAxisData)
        self.myHtml.page().runJavaScript(jscode)   
    
    def setTitle(self, title: str):
        jscode = '''setTitle('{}')'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        self.myHtml.page().runJavaScript("build('{}', '{}', {}, {}, {}); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset), 
                                                                    self.xAxis, self.legends, self.colors))
    def clearData(self):
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        self.myHtml.page().runJavaScript("destroy();")
        pass       

class Pie(object):
    '''
    @param: basehtml: absolute path of html file
    @param: qtcontainer: container to visualize html, only test on QScrollArea now
    '''
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, colors=[], title="") -> None:
        super().__init__()

        self.myHtml = QWebEngineView()
        self.myHtml.loadFinished.connect(self.slotHtmlLoadFinished)

        self.container = qtcontainer
        self.container.setWidget(self.myHtml)

        self.htmlLoadFinished = False
    
        htmlFilename = "file:///{}".format(basehtml)
        self.myHtml.load(QUrl(htmlFilename))

        # self.legend = legend
        self.title = title

        self.colors = colors        

        self.resizeTimer = QTimer()
        self.resizeTimer.setInterval(300)
        self.resizeTimer.timeout.connect(self.slotResize)
        self.resizeTimer.start()
        
        self.width = self.container.width()
        self.height = self.container.height()        

    def slotHtmlLoadFinished(self):
        self.htmlLoadFinished = True
        self.build()
        self.setTitle(self.title)

    def slotResize(self):
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()

    def update(self, data: dict):
        keys = [str(i) for i in data.keys()]
        values = [str(i) for i in data.values()]

        if len(self.colors) < len(keys):
            for i in range(len(self.colors), len(keys)):
                self.colors.append(np.random.choice(default_colors, 1)[0])

        jscode = '''update({},{},{})'''.format(keys, values, self.colors)
        self.myHtml.page().runJavaScript(jscode)
    
    def setTitle(self, title: str):
        jscode = '''setTitle('{}')'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        self.myHtml.page().runJavaScript("build('{}', '{}'); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset)))
    
    def clearData(self):
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        self.myHtml.page().runJavaScript("destroy();")
        pass       