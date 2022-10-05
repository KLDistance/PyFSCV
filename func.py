import sys, os, dill, csv
import random, time
import numpy as np
import pandas as pd
import nifpga
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from PyQt6 import QtCore, QtWidgets, uic, QtGui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QWidget
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot

# this is a test demo for running FSCV on SICM-alpha
# All the process in this program:
# 1. MainUI Process: UI QThread and Message QThread
# 2. BackgroundManager Process: handle signal and params from UI, refactor results from DataRecver to plots
# 2. DataRecver Process: grab data from FPGA FIFO and do basic manipulation
# 3. FileManager Process: store data taken from DataRecver

class CVParams:
    cv_param_low_v = 0.0
    cv_param_high_v = 0.0
    cv_param_vps = 0.0
    cv_param_pos_scan = False
    def __init__(self):
        pass

class CVResults:
    cv_result_volt = np.array([]).astype(float)
    cv_result_curr = np.array([]).astype(float)
    def __init__(self):
        pass

def _FPGAManager(q_list):
    print('FPGAManager PID: ' + str(os.getpid()))
    # FPGA Manager uses queue_list[2]
    # trigger and data params from UI process
    param_dict = {'low_v':0.0, 'high_v':0.0, 'init_v':0.0, 'scan_vps':0.0, 'cycles':0, 'segment_res':0, 'pos_scan':False}
    main_ui_on_running = True
    while main_ui_on_running:
        while True:
            msg = q_list[2].get()
            if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'trigger':
                param_dict = msg['data']
                break
            if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                main_ui_on_running = False
                break
        if not main_ui_on_running:
            break
        # preset containers
        counter = 0
        plot_vec_x = np.array([])
        plot_vec_y = np.array([])
        vec_x = np.array([])
        vec_y = np.array([])
        # preset calculations
        rampw_us = int(abs(param_dict['high_v'] - param_dict['low_v']) / param_dict['segment_res'] / param_dict['scan_vps'] * 1e6)
        # FPGA session
        with nifpga.Session(bitfile='FPGATarget_fscv.lvbitx', resource='RIO0') as session:
            session.reset()
            # obtain controls
            session_LOWV_ctrl = session.registers['LowV']
            session_HIGHV_ctrl = session.registers['HighV']
            session_INITV_ctrl = session.registers['InitV']
            session_CYCLES_ctrl = session.registers['Cycles']
            session_SEGRES_ctrl = session.registers['SegmentResolution']
            session_RAMPW_ctrl = session.registers['Rampw us']
            # obtain indicators
            session_FINISHED_idcr = session.registers['Finished']
            # obtain FIFOs
            session_TTH_FIFO = session.fifos['FIFO']
            # params write
            session_LOWV_ctrl.write(param_dict['low_v'])
            session_HIGHV_ctrl.write(param_dict['high_v'])
            session_INITV_ctrl.write(param_dict['init_v'])
            session_CYCLES_ctrl.write(param_dict['cycles'])
            session_SEGRES_ctrl.write(param_dict['segment_res'])
            session_RAMPW_ctrl.write(rampw_us)
            # trigger FPGA to start
            session.run()
            session_TTH_FIFO.start()
        while True:
            # check notifiers to exit
            fpga_finished = session_FINISHED_idcr.read()
            halt = False
            if not q_list[2].empty():
                msg = q_list[2].get()
                if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'halt':
                    halt = True
                if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                    main_ui_on_running = False
            if fpga_finished or halt or (not main_ui_on_running):
                break
            # data from target-to-host FIFO
            probe_ret = session_TTH_FIFO.read(0, timeout_ms=0)
            if probe_ret.elements_remaining > 0:
                # take out data from FIFO in the multiple of 2
                extract_num = (probe_ret.elements_remaining // 2) * 2
                data_ret = session_TTH_FIFO.read(extract_num, timeout_ms=5000)
                data_arr = np.array(data_ret.data)
                # reshape
                data_reshaped = data_arr((2, extract_num//2))
                vec_x = np.asarray(data_reshaped[0, :])
                vec_y = np.asarray(data_reshaped[1, :])
                plot_vec_x = plot_vec_crop(np.concatenate((plot_vec_x, vec_x)))
                plot_vec_y = plot_vec_crop(np.concatenate((plot_vec_y, vec_y)))
                # send data to parallel processes
                q_list[3].put({'sender':'FPGAManager_Save', 'command':'save', 'data':data_reshaped})
                q_list[0].put({'sender':'FPGAManager_Plot', 'command':'nop', 'data':(plot_vec_x, plot_vec_y)})
            QtCore.QThread.msleep(50)
        if main_ui_on_running:
            break

def plot_vec_crop(arr):
    buff_len = 5000
    if arr.shape[0] > buff_len:
        return arr[arr.shape[0]-buff_len:-1]
    else:
        return arr

def _FileManager(q_list):
    # File Manager uses queue_list[3]
    print('FileManager PID: ' + str(os.getpid()))
    data_path = './data_file.csv'
    with open(data_path, 'a') as csv_handle:
        csv_writer = csv.writer(csv_handle, delimiter=',')
        while True:
            msg = q_list[3].get()
            if msg['sender'] == 'FPGAManager_Save' and msg['command'] == 'save':
                data_recv = msg['data']
            if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                break
            csv_writer.writerows(data_recv.T)
        csv_handle.close()

def _BackgroundManager(q_list):
    print('BackgroundManager PID: ' + str(os.getpid()))
    # BackgroundManager uses queue_list[1]
    # FPGA manager
    fpga_manager_p = Process(target=_FPGAManager, args=(q_list,))
    # file manager
    #file_manager = FileManager()
    #file_manager_p = Process(target=file_manager._run, args=(q_list,))
    fpga_manager_p.start()
    # event loop
    while True:
        msg = q_list[1].get()
        # message categorized
        if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'trigger':
            q_list[2].put(msg)
        if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
            break
        if msg['sender'] == 'FPGAManager_Plot':
            q_list[0].put(msg)
        if msg['sender'] == 'FPGAManager_Save':
            q_list[3].put(msg)
    fpga_manager_p.join()

# msg is dictionary {"sender": "", "command": "nop", "data": ...}

class MyApp_Messenger(QtCore.QThread):
    signal_update_graph = QtCore.pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.message_thread = QtCore.QThread()
        self.moveToThread(self.message_thread)
        self.message_thread.started.connect(self._run)
        # MyApp_Messenger uses queue_list[0]
        self.queue_manager = Manager()
        self.queue_list = []
        for iter in range(4):
            self.queue_list.append(self.queue_manager.Queue())
        # define plot update frames
        self.frame_timer_reset = 0.05
    def thread_start(self):
        self.message_thread.start()
    def _run(self):
        print('MyApp_Messenger PID: ' + str(os.getpid()))
        # Generating lists of queues from manager
        # initialize BackgroundManager process
        self.background_manager_proc = Process(target=_BackgroundManager, args=(self.queue_list,))
        self.background_manager_proc.start()
        # frame_time
        frame_timer = self.frame_timer_reset
        while True:
            tic = time.time()
            msg = self.queue_list[0].get()
            if msg['sender'] == 'FPGAManager_Plot' and frame_timer <= 0:
                self.signal_update_graph.emit(msg['data'])
                frame_timer = self.frame_timer_reset
            if msg['sender'] == 'FileManager':
                pass
            if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                break
            toc = time.time()
            frame_timer -= (toc - tic)
        self.background_manager_proc.join()
        return 0

class MyApp(QWidget):
    # font size
    ui_graph_fontsize = 16
    def __init__(self):
        super().__init__()
        print('MyApp PID: ' + str(os.getpid()))
        uic.loadUi('CV_Form.ui', self)
        # graph initialization
        self.graph_init()
        # message thread
        self.messenger = MyApp_Messenger()
        self.messenger.signal_update_graph.connect(self.update_graph)
        self.messenger.thread_start()
        # connect signal and slot
        self.Btn_Run.clicked.connect(self.on_btn_clicked_RUN)
    def graph_init(self):
        # show grid
        self.CVGraph.showGrid(x=True, y=True)
        # set graph labels
        self.CVGraph.setLabel('left', 'Current', units='A')
        self.CVGraph.setLabel('bottom', 'Potential', units='V')
        # set tick font
        font = QtGui.QFont()
        font.setPixelSize(self.ui_graph_fontsize)
        self.CVGraph.getAxis('left').setStyle(tickFont=font)
        self.CVGraph.getAxis('bottom').setStyle(tickFont=font)
    def update_graph(self, data):
        self.CVGraph.clear()
        self.CVGraph.plot(data[0], data[1])
    def on_btn_clicked_RUN(self):
        low_v = float(self.lineEdit_param_low_v.text())
        high_v = float(self.lineEdit_param_high_v.text())
        init_v = float(self.lineEdit_param_init_v.text())
        scan_vps = float(self.lineEdit_param_vps.text())
        cycles = int(self.lineEdit_param_cycles.text())
        segres = int(self.lineEdit_param_segres.text())
        pos_scan = self.checkBox_param_is_positive_scan.isChecked()
        param_dict = {'low_v':low_v, 'high_v':high_v, 'init_v':init_v, 'scan_vps':scan_vps, 'cycles':cycles, 'segment_res':segres, 'pos_scan':pos_scan}
        self.messenger.queue_list[2].put({'sender':'MyApp_Messenger', 'command':'trigger', 'data':param_dict})
    def on_window_exit(self):
        self.messenger.queue_list[2].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.queue_list[1].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.queue_list[0].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.message_thread.quit()
        self.messenger.message_thread.wait()

