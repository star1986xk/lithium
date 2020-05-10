import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pylab as mpl
from PyQt5.QtCore import pyqtSignal


# 创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    sig_change_size = pyqtSignal(float)
    def __init__(self, width=10, height=10, dpi=100):
        try:
            # 创建一个创建Figure
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            # 在父类中激活Figure窗口
            super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
            # 指定字体,显示中文
            mpl.rcParams['font.sans-serif'] = ['FangSong']
            mpl.rcParams['axes.unicode_minus'] = False

            # 创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
            self.axes = self.fig.add_subplot(111)
        except Exception as e:
            print(e)

    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()  # 竖直滚过的距离
        if angleY > 0:
            self.sig_change_size.emit(-0.1)
        else:  # 滚轮下滚
            self.sig_change_size.emit(0.1)
