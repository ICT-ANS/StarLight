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
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, legends: list, colors=[], symbols=[], title="") -> None:      
        """init method for Line class

        Parameters
        ----------
        basehtml : str
            absolute path of html file
        qtcontainer : QScrollArea
            container to visualize html, only test on QScrollArea now
        legends : list
            legend name
        colors : list, optional
            line color, by default []
        symbols : list, optional
            line symbol, by default []
        title : str, optional
            chater title, by default ""
        """        
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
        """slot function for loadFinished signal triggered by QWebEngineView 
        """               
        self.htmlLoadFinished = True
        self.setTitle(self.title)
        self.build()
    
    def slotResize(self):
        """function for auto resize
        """        
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()

    def update(self, xAxisData: any, yAxisData: list):
        """_summary_

        Parameters
        ----------
        xAxisData : any
            specify xAxisData
        yAxisData : list
            specify yAxisData
        """        
        assert len(self.legends) == len(yAxisData)
        yAxisData = [str(i) for i in yAxisData]
        xAxisData = str(xAxisData)
        jscode = '''update({}, {});'''.format(xAxisData, yAxisData)
        self.myHtml.page().runJavaScript(jscode)   

    def setTitle(self, title: str):
        """set title

        Parameters
        ----------
        title : str
            title name 
        """        
        jscode = '''setTitle('{}');'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        """create Line entry
        """        
        self.myHtml.page().runJavaScript("build('{}', '{}', {}, {}, {}); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset), 
                                                                    self.legends, self.colors, self.symbols))
    def clearData(self):
        """only clear data, you can still call update.
        """        
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        """destroy Line entry, you must call build after destroy.
        """        
        self.myHtml.page().runJavaScript("destroy();")
        pass      

class Bar(object):
    '''
    @param: basehtml: absolute path of html file
    @param: qtcontainer: container to visualize html, only test on QScrollArea now
    @param: legends: list of str for legends
    '''
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, xAxis: list, legends: list, colors=[], title="") -> None:
        """init method for Bar class

        Parameters
        ----------
        basehtml : str
            absolute path of html file
        qtcontainer : QScrollArea
            container to visualize html, only test on QScrollArea now
        xAxis : list
            names for xAxis
        legends : list
            legend name
        colors : list, optional
            Bar color, by default []
        title : str, optional
            chater title, by default""
        """           
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
        """slot function for loadFinished signal triggered by QWebEngineView 
        """            
        self.htmlLoadFinished = True
        self.setTitle(self.title)
        self.build()

    def slotResize(self):
        """function for auto resize
        """          
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()
    
    def update(self, yAxisData: list):
        """update Bar

        Parameters
        ----------
        yAxisData : list
            specify Bar values
        """        
        yAxisData = [str(i) for i in yAxisData]
        jscode = '''update({})'''.format(yAxisData)
        self.myHtml.page().runJavaScript(jscode)   
    
    def setTitle(self, title: str):
        """set title

        Parameters
        ----------
        title : str
            title name 
        """             
        jscode = '''setTitle('{}')'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        """create Bar entry
        """              
        self.myHtml.page().runJavaScript("build('{}', '{}', {}, {}, {}); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset), 
                                                                    self.xAxis, self.legends, self.colors))
    def clearData(self):
        """only clear data, you can still call update.
        """          
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        """destroy Bar entry, you must call build after destroy.
        """          
        self.myHtml.page().runJavaScript("destroy();")
        pass       

class Pie(object):
    def __init__(self, basehtml: str, qtcontainer: QScrollArea, colors=[], title="") -> None:
        """init method for Pie class

        Parameters
        ----------
        basehtml : str
            absolute path of html file
        qtcontainer : QScrollArea
            container to visualize html, only test on QScrollArea now
        colors : list, optional
            Pie color, by default []
        title : str, optional
            title name, by default ""
        """        
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
        """slot function for loadFinished signal triggered by QWebEngineView 
        """         
        self.htmlLoadFinished = True
        self.build()
        self.setTitle(self.title)

    def slotResize(self):
        """function for auto resize
        """             
        if self.htmlLoadFinished:
            if self.width != self.container.width() or self.height != self.container.height():
                self.width  = self.container.width()
                self.height = self.container.height()
                self.build()

    def update(self, data: dict):
        """update Pie

        Parameters
        ----------
        data : dict
            key -> name
            value -> data
        """
        keys = [str(i) for i in data.keys()]
        values = [str(i) for i in data.values()]

        if len(self.colors) < len(keys):
            for i in range(len(self.colors), len(keys)):
                self.colors.append(np.random.choice(default_colors, 1)[0])

        jscode = '''update({},{},{})'''.format(keys, values, self.colors)
        self.myHtml.page().runJavaScript(jscode)
    
    def setTitle(self, title: str):
        """set title

        Parameters
        ----------
        title : str
            title name 
        """            
        jscode = '''setTitle('{}')'''.format(title)
        self.myHtml.page().runJavaScript(jscode)      

    def build(self):
        """create Bar entry
        """         
        self.myHtml.page().runJavaScript("build('{}', '{}'); ".format(int(self.container.width()-offset), 
                                                                    int(self.container.height()-offset)))
    
    def clearData(self):
        """only clear data, you can still call update.
        """          
        self.myHtml.page().runJavaScript("clearData();")
        pass

    def destroy(self):
        """destroy Bar entry, you must call build after destroy.
        """          
        self.myHtml.page().runJavaScript("destroy();")
        pass       