import math
import sys
import datetime
from UI.Ui_mainWin import Ui_Form
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QFileDialog, QScrollArea
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import *
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
import copy
from MyFigure import MyFigure


class lithium(Ui_Form, QWidget):
    def __init__(self, parent=None):
        '''
        程序初始化
        :param parent:
        '''
        super(lithium, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)

        self.radioButton_0.toggled.connect(lambda: self.radio_btn(self.radioButton_0))
        self.radioButton_1.toggled.connect(lambda: self.radio_btn(self.radioButton_1))
        self.radioButton_2.toggled.connect(lambda: self.radio_btn(self.radioButton_2))
        self.radioButton_3.toggled.connect(lambda: self.radio_btn(self.radioButton_3))

        self.pushButton_inExcel.clicked.connect(self.inExcel)
        self.pushButton_run.clicked.connect(self.run)
        self.pushButton_outExcel.clicked.connect(self.outExcel)
        self.pushButton_outImage.clicked.connect(self.outImage)
        self.pushButton_up.clicked.connect(self.up)
        self.pushButton_down.clicked.connect(self.down)

    def radio_btn(self, btn):
        index = int(btn.objectName()[-1])
        self.stackedWidget_2.setCurrentIndex(index)

    def inExcel(self):
        '''
        导入excel路径
        :return:
        '''
        filename, _ = QFileDialog.getOpenFileName(self, '选取文件', './', 'Excel (*.xlsx; *.xls)')
        if filename:
            self.lineEdit_inExcel.setText(filename)

    def outExcel(self):
        '''
        保存excel
        :return:
        '''
        try:
            if self.datas:
                for k, v in self.datas.items():
                    fileName, ok = QFileDialog.getSaveFileName(None, "文件保存", "./" + k, "csv (*.csv)")
                    v.to_csv(fileName, encoding='utf_8_sig')
        except Exception as e:
            print(e)

    def outImage(self):
        '''
        保存图片
        :return:
        '''
        index = self.stackedWidget.currentIndex()
        if index >= 0:
            fileName, ok = QFileDialog.getSaveFileName(None, "文件保存", "./", "PNG (*.PNG)")
            self.fig_list[index].fig.savefig(fileName)

    def up(self):
        '''
        上一页
        :return:
        '''
        index = self.stackedWidget.currentIndex()
        self.stackedWidget.setCurrentIndex(index - 1)
        self.label_page.setText(str(self.stackedWidget.currentIndex() + 1) + '/' + str(self.page_count))

    def down(self):
        '''
        下一页
        :return:
        '''
        index = self.stackedWidget.currentIndex()
        self.stackedWidget.setCurrentIndex(index + 1)
        self.label_page.setText(str(self.stackedWidget.currentIndex() + 1) + '/' + str(self.page_count))

    def run(self):
        path = self.lineEdit_inExcel.text().strip()
        if path:
            self.datas = {}
            self.fig_list = []
            self.textBrowser.clear()
            index = self.stackedWidget_2.currentIndex()
            if index == 0:
                self.func1(path)
            elif index == 1:
                self.func2(path)
            elif index ==2:
                self.func3(path)
            elif index ==3:
                self.func4(path)

    def loadImg(self):
        # 清空stackedWidget
        for i in range(self.stackedWidget.count(), -1, -1):
            widget = self.stackedWidget.widget(i)
            self.stackedWidget.removeWidget(widget)
        # 写入图片
        for fig in self.fig_list:
            page = QWidget()
            verticalLayout = QVBoxLayout(page)
            w = QWidget()
            verticalLayout1 = QVBoxLayout(w)
            verticalLayout1.addWidget(fig)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(w)
            verticalLayout.addWidget(scroll)
            self.stackedWidget.addWidget(page)
        self.page_now = self.stackedWidget.currentIndex() + 1
        self.page_count = len(self.fig_list)
        self.label_page.setText(str(self.page_now) + '/' + str(self.page_count))

    def func1(self, file_path):
        file = pd.ExcelFile(file_path)
        df = pd.read_excel(file)
        data = pd.DataFrame(df)
        data.dropna(axis=0, how='any', inplace=True)  # 删除空白行
        data = data.reset_index(drop=True)  # 重置索引
        column_headers = list(data.columns.values)  # 读取文件第一行的列标签
        record_time = data['时间']
        soc = data['动力电池剩余电量SOC']
        totall_voltage = data['动力电池内部总电压V1']
        current = data['动力电池充/放电电流']
        R1 = data['动力电池正极对GND绝缘电阻']
        R2 = data['动力电池负极对GND绝缘电阻']
        alarm_level = data['整车最高报警等级']
        charging_state = data['充电状态']
        fault_batteries_number = data['动力电池故障总数']
        voltage_data = np.array(data.iloc[:, 94:178])  # 1-84号单体电池电压
        tempreture_data = np.array(data.iloc[:, 178:206])  # 1-28号温度检测点温度
        (len_x, len_y) = voltage_data.shape
        (len_p, len_q) = tempreture_data.shape

        # 时间格式转换
        def datetimeToTimeNum(time_array):
            time_num = list()
            for time_str in time_array:
                temp_date = datetime.datetime.strptime(time_str, r"%Y/%m/%d %H:%M:%S.%f.")
                temp_time = temp_date.timetuple()
                temp_num = int(time.mktime(temp_time))
                time_num.append(temp_num)
            return np.array(time_num)

        t = datetimeToTimeNum(record_time)

        Voltage_data = copy.deepcopy(voltage_data)
        for i in range(len_x):
            if Voltage_data[i, :].all() == 0 or 1.44:
                Voltage_data[i, :] == 'nan'
        Tempreture_data = copy.deepcopy(tempreture_data)
        for j in range(len_p):
            if Tempreture_data[j, :].all() == 0:
                Tempreture_data[j, :] == 'nan'
        list1 = []
        #  整车最高报警等级
        alarm_level_max = alarm_level.max()
        list1.append(alarm_level_max)
        # print('整车最高报警等级为：')
        # print(alarm_level_max)
        #  温度采集电路故障
        for i in range(len_p):
            for j in range(len_q):
                if Tempreture_data[i, j] == 0:
                    list1.append('温度采集电路故障')
        #  单体电压采集电路故障
        for i in range(len_x):
            for j in range(len_y):
                if Voltage_data[i, j] == 0 or 1.44:
                    list1.append('单体电压采集电路故障')
        #   绝缘电阻低故障
        for i in range(len(t)):
            if R1[i] < 200 or R2[i] < 200:
                list1.append('绝缘电阻低故障')
        #  总电压欠压故障
        for i in range(len(t)):
            if totall_voltage[i] == 0 and totall_voltage[i + 1] == 0:
                list1.append('总电压欠压故障')
        #   SOC太低故障
        for i in range(len(t)):
            if soc[i] == 0 and current[i] != 0:
                list1.append('SOC太低故障')
        #   SOC跳变故障
        for i in range(len(t) - 1):
            if abs(soc[i] - soc[i + 1]) > 10:
                list1.append('SOC跳变故障')
        fault_analysis_result = []
        for i in list1:
            if not i in fault_analysis_result:
                fault_analysis_result.append(i)
        fault_analysis_result = pd.DataFrame(fault_analysis_result)
        fault_analysis_result = fault_analysis_result.T
        fault_analysis_result.rename(
            columns={0: '整车最高报警等级', 1: '故障1', 2: '故障2', 3: '故障3', 4: '故障4', 5: '故障5', 6: '故障6'}, inplace=True)
        print(fault_analysis_result.columns)
        print(fault_analysis_result.values)
        self.textBrowser.append('    '.join(['{:<30}'.format(str(li)) for li in fault_analysis_result.columns]))
        for list1 in fault_analysis_result.values:
            self.textBrowser.append('    '.join(['{:<30}'.format(str(li)) for li in list1]))
        self.datas['整车故障分析'] = fault_analysis_result

    def func2(self, file_path):
        file = pd.ExcelFile(file_path)
        df = pd.read_excel(file)
        data = pd.DataFrame(df)
        data.dropna(axis=0, how='any', inplace=True)  # 删除空白行
        data = data.reset_index(drop=True)  # 重置索引
        column_headers = list(data.columns.values)  # 读取文件第一行的列标签
        record_time = data['时间']
        soc = data['动力电池剩余电量SOC']
        current = data['动力电池充/放电电流']
        voltage_data = np.array(data.iloc[:, 94:178])  # 1-84号单体电池电压
        (len_x, len_y) = voltage_data.shape

        # 时间格式转换
        def datetimeToTimeNum(time_array):
            time_num = list()
            for time_str in time_array:
                temp_date = datetime.datetime.strptime(time_str, r"%Y/%m/%d %H:%M:%S.%f.")
                temp_time = temp_date.timetuple()
                temp_num = int(time.mktime(temp_time))
                time_num.append(temp_num)
            return np.array(time_num)

        t = datetimeToTimeNum(record_time)

        # 充电段提取
        def Charge_period(soc, current):
            start_index = 0
            stop_index = 0
            point_start = []
            point_stop = []
            part_number = 0
            flag = 0
            for i in np.arange(1, soc.size, 1):
                if i == 1:
                    if (flag == 0 and soc[i] >= soc[i - 1] and current[i] < 0):
                        start_index = i
                        flag = 1
                elif i == len(current + 1):
                    if (flag == 1 and soc[i] <= soc[i - 1] and current[i - 1] > 0):
                        stop_index = i - 1
                        flag = 2
                else:
                    if (flag == 0 and current[i] < 0 and current[i - 1] >= 0):
                        start_index = i
                        flag = 1
                    elif (flag == 1 and current[i] >= 0 and current[i - 1] < 0):
                        stop_index = i
                        flag = 2
                if flag == 2:
                    if (soc[stop_index] > soc[start_index] and (stop_index - start_index) > 10):
                        point_start.append(start_index)
                        point_stop.append(stop_index)
                        part_number = part_number + 1
                    else:
                        start_index = 0
                        stop_index = 0
                    flag = 0
            return point_start, point_stop, part_number

        start_index, stop_index, charge_number = Charge_period(soc, current)
        #   单体电压线性插值
        voltage_data_all = copy.deepcopy(voltage_data)
        for i in range(len_x):
            for j in range(len_y):
                if voltage_data_all[i, j] == 0 or voltage_data_all[i, j] == 1.44:
                    voltage_data_all[i, j] = 'nan'
                elif voltage_data_all[i, j] > 4.2:
                    voltage_data_all[i, j] = voltage_data_all[i - 1, j]
        voltage_data_qualified = copy.deepcopy(voltage_data_all)
        for q in range(len_y):
            for p in range(1, len_x - 1):
                if voltage_data_qualified[p, q] == 'nan':
                    p = p + 1
                elif voltage_data_qualified[p, q] == voltage_data_qualified[p + 1, q]:
                    voltage_data_qualified[p, q] = 'nan'
        voltage_data_fill = copy.deepcopy(voltage_data_qualified)
        #  所有单体电压数据后期都用最后一个可用值补上
        for j in range(len_y):
            for i in range(1, len_x):
                if np.isnan(voltage_data_fill[len_x - i, j]) == 0:
                    voltage_data_fill[len_x - i:len_x, j] = voltage_data_fill[len_x - i, j]
                    break
        #  从头开始所有数据都用第一个可用值补上
        for j in range(len_y):
            for i in range(len_x):
                if np.isnan(voltage_data_fill[i, j]) == 0:
                    voltage_data_fill[0:i, j] = voltage_data_fill[i, j]
                    break
        #  所有中间数据都用相邻的值进行插值
        for j in range(len_y):
            for i in range(1, len_x - 1):
                c = 1
                if np.isnan(voltage_data_fill[i, j]):
                    k = i + c - 1
                    while np.isnan(voltage_data_fill[i, j]):
                        i = i + 1
                    voltage_data_fill[k:i, j] = np.linspace(voltage_data_fill[k - 1, j], voltage_data_fill[i, j], i - k)
        #   电池单体电压不均衡
        for i in range(len_x):
            Voltage_data_max = np.nanmax(voltage_data_all[i, :])
            Voltage_data_min = np.nanmin(voltage_data_all[i, :])
            Voltage_diff = Voltage_data_max - Voltage_data_min
            if Voltage_diff > 1:
                self.textBrowser.append('电池单体电压不均衡故障')
                print('电池单体电压不均衡故障')
                break
        #   单体欠压故障
        for j in range(len_y):
            for i in range(len_x):
                if voltage_data_all[i, j] < 2.5:
                    self.textBrowser.append('单体欠压故障,故障单体电池编号为：{}'.format(j + 1))
                    print('单体欠压故障,故障单体电池编号为：{}'.format(j + 1))
                    break
        # 添加图片列表
        self.fig_list = []

        #   子功能1：电压差及变化率
        if self.checkBox_21.isChecked():
            Voltage_data_diff = np.zeros((len_x - 1, len_y))
            Voltage_change_rate = np.zeros((len_x - 1, len_y))
            time_diff = np.zeros((len(t) - 1, 1))
            for i in range(len_y):
                time_diff = np.diff(t)
                Voltage_data_diff[:, i] = np.diff(voltage_data_fill[:, i])
                Voltage_change_rate[:, i] = Voltage_data_diff[:, i] / time_diff

            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(Voltage_data_diff[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('电压差')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('电压差(V)', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(Voltage_change_rate[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('电压变化率')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('电压变化率', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            Voltage_data_diff = pd.DataFrame(Voltage_data_diff)
            Voltage_change_rate = pd.DataFrame(Voltage_change_rate)
            self.datas['电压差'] = Voltage_data_diff
            self.datas['电压变化率'] = Voltage_change_rate

        #   子功能2：最大值、最小值和极差
        if self.checkBox_22.isChecked():
            voltage_statistical_result = np.zeros((len(t), 5))
            for i in range(len(t)):
                voltage_statistical_result[i, 0] = voltage_data_fill[i, :].max()
                voltage_statistical_result[i, 1] = voltage_data_fill[i, :].min()
                voltage_statistical_result[i, 2] = voltage_statistical_result[i, 0] - voltage_statistical_result[i, 1]
                voltage_statistical_result[i, 3] = voltage_data_fill[i, :].mean()
                voltage_statistical_result[i, 4] = voltage_data_fill[i, :].std()

            MyFig = MyFigure()
            MyFig.axes.plot(voltage_statistical_result[:, 0], color='red', label='最大值', linewidth=2, linestyle='-')
            MyFig.axes.plot(voltage_statistical_result[:, 1], color='blue', label='最小值', linewidth=2, linestyle='-')
            MyFig.axes.legend(loc='lower left')
            MyFig.axes.set_title('电压最大值最小值')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('电压（V）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            MyFig.axes.plot(voltage_statistical_result[:, 2], color='black', label='极  差', linewidth=2, linestyle='-')
            MyFig.axes.legend(loc='upper left')
            MyFig.axes.set_title('电压极差')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('电压（V）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            voltage_statistical_result = pd.DataFrame(voltage_statistical_result)
            voltage_statistical_result.rename(columns={0: '最大值', 1: '最小值', 2: '极差', 3: '平均值', 4: '标准差'}, inplace=True)
            self.datas['电压统计特征值'] = voltage_statistical_result

        #    子功能3：电压偏离度
        if self.checkBox_23.isChecked():
            voltage_data_dev = np.zeros((len(t), len_y))
            voltage_data_dev_degree = np.zeros((len(t), len_y))
            for i in range(len(t)):
                for j in range(len_y):
                    voltage_data_dev[i, j] = voltage_data_fill[i, j] - voltage_data_fill[i, :].mean()
                    voltage_data_dev_degree[i, j] = abs(
                        voltage_data_fill[i, j] - voltage_data_fill[i, :].mean()) / voltage_data_fill[i, :].mean()
            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(voltage_data_dev_degree[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('电压偏离度')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('偏离度', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)
            voltage_data_dev_degree = pd.DataFrame(voltage_data_dev_degree)
            self.datas['电压偏离度'] = voltage_data_dev_degree

        #   子功能4：充电起始电压和充电结束电压及排名
        if self.checkBox_24.isChecked():
            charge_start_voltage = np.zeros((len_y, charge_number))
            charge_stop_voltage = np.zeros((len_y, charge_number))
            charge_start_voltage_rank = np.zeros((len_y, charge_number))
            charge_stop_voltage_rank = np.zeros((len_y, charge_number))
            for n in range(charge_number):
                a = start_index[n]
                b = stop_index[n]
                charge_voltage = voltage_data_fill[a:b, :]
                charge_start_voltage[:, n] = charge_voltage[0, :].T
                charge_stop_voltage[:, n] = charge_voltage[-1, :].T
                charge_start_voltage_rank[:, n] = np.argsort(charge_start_voltage[:, n]) + 1
                charge_stop_voltage_rank[:, n] = np.argsort(charge_stop_voltage[:, n]) + 1
            df1 = pd.DataFrame(charge_start_voltage)
            df2 = pd.DataFrame(charge_start_voltage_rank)
            df3 = pd.DataFrame(charge_stop_voltage)
            df4 = pd.DataFrame(charge_stop_voltage_rank)
            writer = pd.ExcelWriter('充电起始和结束电压及排名.xlsx')
            df1.to_excel(writer, "充电起始电压")
            df2.to_excel(writer, "充电起始电压排名")
            df3.to_excel(writer, "充电结束电压")
            df4.to_excel(writer, "充电结束电压排名")
            writer.save()


            MyFig = MyFigure()
            MyFig.axes.boxplot(df1,sym="o", whis=1.5, medianprops={'color': 'green'}, boxprops=dict(color="blue"),
                        flierprops={'color': 'red'})
            MyFig.axes.set_title('充电起始电压箱型图')
            MyFig.axes.set_xlabel('充电次数', fontsize=14)
            MyFig.axes.set_ylabel('充电起始电压（V）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            MyFig.axes.boxplot(df3,sym="o", whis=1.5, medianprops={'color': 'green'}, boxprops=dict(color="blue"),
                        flierprops={'color': 'red'})
            MyFig.axes.set_title('充电结束电压箱型图')
            MyFig.axes.set_xlabel('充电次数', fontsize=14)
            MyFig.axes.set_ylabel('充电结束电压（V）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            for n in range(charge_number):
                MyFig.axes.plot(charge_start_voltage_rank[n, :], linewidth=1, linestyle='-')
            MyFig.axes.set_xticks(range(0, 90, 10))
            MyFig.axes.set_xlabel('单体电池编号', fontsize=14)
            MyFig.axes.set_ylabel('充电起始时刻电压排名', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            for n in range(charge_number):
                MyFig.axes.plot(charge_stop_voltage_rank[n, :], linewidth=1, linestyle='-')
            MyFig.axes.set_xticks(range(0, 90, 10))
            MyFig.axes.set_xlabel('单体电池编号', fontsize=14)
            MyFig.axes.set_ylabel('充电结束时刻电压排名', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)
        self.loadImg()

    def func3(self,file_path):
        file = pd.ExcelFile(file_path)
        df = pd.read_excel(file)
        data = pd.DataFrame(df)
        data.dropna(axis=0, how='any', inplace=True)  # 删除空白行
        data = data.reset_index(drop=True)  # 重置索引
        column_headers = list(data.columns.values)  # 读取文件第一行的列标签
        record_time = data['时间']
        soc = data['动力电池剩余电量SOC']
        current = data['动力电池充/放电电流']
        tempreture_data = np.array(data.iloc[:, 178:206])  # 1-28号温度检测点温度
        (len_x, len_y) = tempreture_data.shape

        # 时间格式转换
        def datetimeToTimeNum(time_array):
            time_num = list()
            for time_str in time_array:
                temp_date = datetime.datetime.strptime(time_str, r"%Y/%m/%d %H:%M:%S.%f.")
                temp_time = temp_date.timetuple()
                temp_num = int(time.mktime(temp_time))
                time_num.append(temp_num)
            return np.array(time_num)

        t = datetimeToTimeNum(record_time)

        Tempreture_data_all = copy.deepcopy(tempreture_data)
        for i in range(len_x):
            for j in range(len_y):
                if Tempreture_data_all[i, j] == 0 or Tempreture_data_all[i, j] == 130:
                    Tempreture_data_all[i, j] = 'nan'
        Tempreture_data_qualified = copy.deepcopy(Tempreture_data_all)
        for n in range(len_y):
            for m in range(1, len(t) - 1):
                if Tempreture_data_qualified[m, n] == 'nan':
                    m = m + 1

        Tempreture_data_fill = copy.deepcopy(Tempreture_data_qualified)
        #  所有单体电压数据后期都用最后一个可用值补上
        for j in range(len_y):
            for i in range(1, len(t)):
                if np.isnan(Tempreture_data_fill[len(t) - i, j]) == 0:
                    Tempreture_data_fill[len(t) - i:len(t), j] = Tempreture_data_fill[len(t) - i, j]
                    break
        #  从头开始所有数据都用第一个可用值补上
        for j in range(len_y):
            for i in range(len(t)):
                if np.isnan(Tempreture_data_fill[i, j]) == 0:
                    Tempreture_data_fill[0:i, j] = Tempreture_data_fill[i, j]
                    break
        #  所有中间数据都用相邻的值进行插值
        for j in range(len_y):
            for i in range(1, len(t) - 1):
                c = 1
                if np.isnan(Tempreture_data_fill[i, j]):
                    k = i + c - 1
                    while np.isnan(Tempreture_data_fill[i, j]):
                        i = i + 1
                    Tempreture_data_fill[k:i, j] = np.linspace(Tempreture_data_fill[k - 1, j],
                                                               Tempreture_data_fill[i, j], i - k)
        #   温度不均衡故障
        for i in range(len(t)):
            Tempreture_data_max = np.nanmax(Tempreture_data_fill[i, :])
            Tempreture_data_min = np.nanmin(Tempreture_data_fill[i, :])
            Tempreture_diff = Tempreture_data_max - Tempreture_data_min
            if Tempreture_diff > 18:
                self.textBrowser.append('温度不均衡故障')
                print('温度不均衡故障')
                break
        #   温度过高故障
        for j in range(len_y):
            for i in range(len_x):
                if Tempreture_data_fill[i, j] > 65:
                    self.textBrowser.append('温度过高故障，故障温度检测点编号为：{}'.format(j + 1))
                    print('温度过高故障，故障温度检测点编号为：{}'.format(j + 1))
                    break

        # 添加图片列表
        self.fig_list = []

        #   子功能1：温度差及变化率
        if self.checkBox_31.isChecked():
            Tempreture_data_diff = np.zeros((len_x - 1, len_y))
            Tempreture_change_rate = np.zeros((len_x - 1, len_y))
            time_diff = np.zeros((len(t) - 1, 1))
            for i in range(len_y):
                time_diff = np.diff(t)
                Tempreture_data_diff[:, i] = np.diff(Tempreture_data_fill[:, i])
                Tempreture_change_rate[:, i] = Tempreture_data_diff[:, i] / time_diff

            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(Tempreture_data_diff[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('温度差')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('温度差(℃)', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(Tempreture_change_rate[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('温度变化率')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('温度变化率', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            Tempreture_data_diff = pd.DataFrame(Tempreture_data_diff)
            Tempreture_change_rate = pd.DataFrame(Tempreture_change_rate)
            self.datas['温度差'] = Tempreture_data_diff
            self.datas['温度变化率'] = Tempreture_change_rate

        #    子功能2：最大值、最小值和极差
        if self.checkBox_32.isChecked():
            Tempreture_statistical_result = np.zeros((len(t), 5))
            for i in range(len(t)):
                Tempreture_statistical_result[i, 0] = Tempreture_data_fill[i, :].max()
                Tempreture_statistical_result[i, 1] = Tempreture_data_fill[i, :].min()
                Tempreture_statistical_result[i, 2] = Tempreture_statistical_result[i, 0] - Tempreture_statistical_result[
                    i, 1]
                Tempreture_statistical_result[i, 3] = Tempreture_data_fill[i, :].mean()
                Tempreture_statistical_result[i, 4] = Tempreture_data_fill[i, :].std()
            MyFig = MyFigure()
            MyFig.axes.plot(Tempreture_statistical_result[:, 0], color='red', label='最大值', linewidth=2, linestyle='-')
            MyFig.axes.plot(Tempreture_statistical_result[:, 1], color='blue', label='最小值', linewidth=2, linestyle='-')
            MyFig.axes.legend(loc='upper left')
            MyFig.axes.set_title('温度最大值最小值')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('温度（℃）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            MyFig = MyFigure()
            MyFig.axes.plot(Tempreture_statistical_result[:, 2], color='black', label='极  差', linewidth=2, linestyle='-')
            MyFig.axes.legend(loc='upper left')
            MyFig.axes.set_title('温度极差')
            MyFig.axes.set_xlabel('采样数', fontsize=14)
            MyFig.axes.set_ylabel('温度（℃）', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            Tempreture_statistical_result = pd.DataFrame(Tempreture_statistical_result)
            Tempreture_statistical_result.rename(columns={0: '最大值', 1: '最小值', 2: '极差', 3: '平均值', 4: '标准差'}, inplace=True)
            self.datas['温度统计特征值'] = Tempreture_statistical_result

        self.loadImg()

    def func4(self,file_path):
        file = pd.ExcelFile(file_path)
        dd = pd.read_excel(file)
        data = pd.DataFrame(dd)
        data.dropna(axis=0, how='any', inplace=True)  # 删除空白行
        data = data.reset_index(drop=True)  # 重置索引

        record_time = np.array(data.iloc[:, 1])  # 运行时间
        soc = np.array(data.iloc[:, 24])  # 动力电池剩余电量SOC
        current = np.array(data.iloc[:, 29])  # 动力电池充放电电流
        cell_voltages = np.array(data.iloc[:, 94:178])  # 1-84号单体电池电压
        (len_x, len_y) = cell_voltages.shape

        # 时间格式转换
        def datetimeToTimeNum(time_array):
            time_num = list()
            for time_str in time_array:
                temp_date = datetime.datetime.strptime(time_str, r"%Y/%m/%d %H:%M:%S.%f.")
                temp_time = temp_date.timetuple()
                temp_num = int(time.mktime(temp_time))
                time_num.append(temp_num)
            return np.array(time_num)

        t = datetimeToTimeNum(record_time)

        # 充电段提取
        def Charge_period(soc, current):
            start_index = 0
            stop_index = 0
            point_start = []
            point_stop = []
            part_number = 0
            flag = 0
            for i in np.arange(1, soc.size, 1):
                if i == 1:
                    if (flag == 0 and soc[i] >= soc[i - 1] and current[i] < 0):
                        start_index = i
                        flag = 1
                elif i == len(current + 1):
                    if (flag == 1 and soc[i] <= soc[i - 1] and current[i - 1] > 0):
                        stop_index = i - 1
                        flag = 2
                else:
                    if (flag == 0 and current[i] < 0 and current[i - 1] >= 0):
                        start_index = i
                        flag = 1
                    elif (flag == 1 and current[i] >= 0 and current[i - 1] < 0):
                        stop_index = i
                        flag = 2
                if flag == 2:
                    if (soc[stop_index] > soc[start_index] and (stop_index - start_index) > 10):
                        point_start.append(start_index)
                        point_stop.append(stop_index)
                        part_number = part_number + 1
                    else:
                        start_index = 0
                        stop_index = 0
                    flag = 0
            return point_start, point_stop, part_number

        start_index, stop_index, charge_number = Charge_period(soc, current)
        #   单体电压线性插值
        voltage_data_all = copy.deepcopy(cell_voltages)
        for i in range(len_x):
            for j in range(len_y):
                if voltage_data_all[i, j] == 0 or voltage_data_all[i, j] == 1.44:
                    voltage_data_all[i, j] = 'nan'
                elif voltage_data_all[i, j] > 4.2:
                    voltage_data_all[i, j] = voltage_data_all[i - 1, j]
        voltage_data_qualified = copy.deepcopy(voltage_data_all)
        for q in range(len_y):
            for p in range(1, len_x - 1):
                if voltage_data_qualified[p, q] == 'nan':
                    p = p + 1
                elif voltage_data_qualified[p, q] == voltage_data_qualified[p + 1, q]:
                    voltage_data_qualified[p, q] = 'nan'

        voltage_data_fill = copy.deepcopy(voltage_data_qualified)
        #  所有单体电压数据后期都用最后一个可用值补上
        for j in range(len_y):
            for i in range(1, len_x):
                if np.isnan(voltage_data_fill[len_x - i, j]) == 0:
                    voltage_data_fill[len_x - i:len_x, j] = voltage_data_fill[len_x - i, j]
                    break
        #  从头开始所有数据都用第一个可用值补上
        for j in range(len_y):
            for i in range(len_x):
                if np.isnan(voltage_data_fill[i, j]) == 0:
                    voltage_data_fill[0:i, j] = voltage_data_fill[i, j]
                    break
        #  所有中间数据都用相邻的值进行插值
        for j in range(len_y):
            for i in range(1, len_x - 1):
                c = 1
                if np.isnan(voltage_data_fill[i, j]):
                    k = i + c - 1
                    while np.isnan(voltage_data_fill[i, j]):
                        i = i + 1
                    voltage_data_fill[k:i, j] = np.linspace(voltage_data_fill[k - 1, j], voltage_data_fill[i, j], i - k)

        # 添加图片列表
        self.fig_list = []

        #子功能1
        if self.checkBox_41.isChecked():
            n = charge_number - 1
            a = start_index[n]
            b = stop_index[n]
            charge_data = np.zeros((b - a, len_y))
            for i in range(len_y):
                charge_data[:, i] = voltage_data_fill[a:b, i]
            (len_x, len_y) = charge_data.shape
            MyFig = MyFigure()
            for k in range(len_y):
                MyFig.axes.plot(charge_data[:, k], linewidth=1, linestyle='-')
            MyFig.axes.set_title('充电电压')
            MyFig.axes.set_xlabel('时间（10s）', fontsize=14)
            MyFig.axes.set_ylabel('电压(V)', fontsize=14)
            MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
            self.fig_list.append(MyFig)

            lt_1 = []
            lt_2 = []
            lt_3 = []
            Dist_max_data = pd.DataFrame()
            for i in range(len_y - 1):
                for j in range(i + 1, len_y):
                    if i >= j:
                        break
                    else:
                        Voltage_Curve_1 = charge_data[:, i]
                        Voltage_Curve_2 = charge_data[:, j]
                        d = 0
                        for k in range(len_x):
                            d0 = pow((Voltage_Curve_1[k] - Voltage_Curve_2[k]), 2)
                            d += d0
                        Curve_dist = pow(d, 0.5)
                        lt_1.append(Curve_dist)
                        lt_2.append(i + 1)
                        lt_3.append(j + 1)
                    result = [lt_1, lt_2, lt_3]
                    result = pd.DataFrame(result)
                    result = result.T
                    result.rename(columns={0: '欧氏距离', 1: '单体1', 2: '单体2'}, inplace=True)
                    Result = result.sort_values(by='欧氏距离', ascending=False)
                    Result = Result.reset_index(drop=True)
                    # Result.to_csv('电压曲线距离' + '.csv', encoding='utf_8_sig')
                max_data = Result.iloc[0:5]
            Dist_max_data = Dist_max_data.append(max_data)
            self.textBrowser.append('    '.join(['{:<30}'.format(str(li)) for li in Dist_max_data.columns]))
            for list1 in Dist_max_data.values:
                self.textBrowser.append('    '.join(['{:<30}'.format(str(li)) for li in list1]))
            self.datas['电压曲线距离'] = Dist_max_data

        #子功能2
        elif self.checkBox_42.isChecked():
            # 获取任务密度，取第5邻域，阈值为2（LOF大于2认为是离群值）
            def localoutlierfactor(data, predict, k):
                from sklearn.neighbors import LocalOutlierFactor
                clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
                clf.fit(data)
                # 记录 k 邻域距离
                predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
                # 记录 LOF 离群因子，做相反数处理
                predict['local outlier factor'] = -clf._decision_function(predict.iloc[:, :-1])
                return predict

            def plot_lof(result, method):
                MyFig = MyFigure()
                MyFig.axes.scatter(result[result['local outlier factor'] > method].index + 1,
                            result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                            marker='.', alpha=None,label='离群点')
                MyFig.axes.scatter(result[result['local outlier factor'] <= method].index + 1,
                            result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                            marker='.', alpha=None, label='正常点')
                MyFig.axes.hlines(method, -2, 2 + max(result.index), linestyles='--')
                MyFig.axes.legend(loc='upper right')
                MyFig.axes.set_xlim(-2, 2 + max(result.index))
                MyFig.axes.set_title('LOF局部离群点检测')
                MyFig.axes.set_xlabel('单体电池编号', fontsize=14)
                MyFig.axes.set_ylabel('局部离群因子', fontsize=14)
                MyFig.axes.tick_params(axis='both', which='major', direction='in', labelsize=12)
                self.fig_list.append(MyFig)

            def lof(data, predict=None, k=5, method=2, plot=True):
                import pandas as pd
                # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
                try:
                    if predict == None:
                        predict = data.copy()
                except Exception:
                    pass
                predict = pd.DataFrame(predict)
                # 计算 LOF 离群因子
                predict = localoutlierfactor(data, predict, k)
                if plot == True:
                    plot_lof(predict, method)
                # 根据阈值划分离群点与正常点
                outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
                inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
                return outliers, inliers

            n = charge_number - 1
            a = start_index[n]
            b = stop_index[n]
            charge_data = voltage_data_fill[a:b, :]
            charge_data_symbol = np.zeros((2, len_y))
            for i in range(len_y):
                charge_data_symbol[0, i] = charge_data[:, i].mean()
                charge_data_symbol[1, i] = charge_data[:, i].std()
            data = charge_data_symbol.T
            A = np.array(data[:, 0])
            B = np.array(data[:, 1])
            C = list(zip(A, B))
            outliers1, inliers1 = lof(C, k=5, method=2)
        self.loadImg()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = lithium()
    win.show()
    sys.exit(app.exec_())
