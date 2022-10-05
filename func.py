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
# 3. DataRecver Process: grab data from FPGA FIFO and do basic manipulation
# 4. FileManager Process: store data taken from DataRecver

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

#_global_cond_FPGAManager = threading.Condition()
#_global_task_halt_FPGAManager = False

def _FPGAManager_Watchdog(q_list):
    pass

def _FPGAManager(q_list):
    print('FPGAManager PID: ' + str(os.getpid()))
    #hThread_Watchdog = threading.Thread(target = _FPGAManager_Watchdog, args=(q_list,))
    # FPGA Manager uses queue_list[2]
    # trigger and data params from UI process
    param_dict = {'low_v':0.0, 'high_v':0.0, 'init_v':0.0, 'scan_vps':0.0, 'cycles':0, 'segment_res':0, 'pos_scan':False}
    main_ui_on_running = True
    while main_ui_on_running:
        while True:
            print('session ready')
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
        # data_buffer
        data_buffer = np.empty((1,2), dtype=float)
        is_buffer_nonempty = False
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
            session_STOP_ctrl = session.registers['stop']
            # obtain indicators
            session_FINISHED_idcr = session.registers['Finished']
            # obtain FIFOs
            session_TTH_FIFO = session.fifos['FIFO']
            # params write
            session_LOWV_ctrl.write(param_dict['low_v']*5)
            session_HIGHV_ctrl.write(param_dict['high_v']*5)
            session_INITV_ctrl.write(param_dict['init_v']*5)
            session_CYCLES_ctrl.write(param_dict['cycles'])
            session_SEGRES_ctrl.write(param_dict['segment_res'])
            session_RAMPW_ctrl.write(rampw_us)
            # trigger FPGA to start
            session.run()
            session_TTH_FIFO.start()
            while True:
                halt = False
                # check notifiers to exit
                fpga_finished = session_FINISHED_idcr.read()
                if not q_list[2].empty():
                    msg = q_list[2].get()
                    if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'halt':
                        halt = True
                    if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                        main_ui_on_running = False
                # data from target-to-host FIFO
                probe_ret = session_TTH_FIFO.read(0, timeout_ms=0)
                if probe_ret.elements_remaining > 0:
                    # take out data from FIFO in the multiple of 2
                    extract_num = (probe_ret.elements_remaining // 2) * 2
                    data_ret = session_TTH_FIFO.read(extract_num, timeout_ms=5000)
                    data_arr = np.asarray(data_ret.data, dtype='float')
                    # reshape
                    data_reshaped = np.reshape(data_arr, (extract_num//2, 2))
                    data_reshaped[:, 0] /= 5.0
                    data_reshaped[:, 1] *= 1e-9
                    plot_vec_x = plot_vec_crop(np.concatenate((plot_vec_x, np.asarray(data_reshaped[:, 0]))))
                    plot_vec_y = plot_vec_crop(np.concatenate((plot_vec_y, np.asarray(data_reshaped[:, 1]))))
                    data_buffer = np.concatenate((data_buffer, data_reshaped))
                    is_buffer_nonempty = True
                    # send data to parallel processes
                    q_list[0].put({'sender':'FPGAManager', 'command':'draw', 'data':(plot_vec_x, plot_vec_y)})
                    # send large data chunks instead of small ones to the pipe
                    if data_buffer.nbytes > 65536*2:
                        data_buffer = np.delete(data_buffer, obj=0, axis=0)
                        q_list[3].put({'sender':'FPGAManager', 'command':'save', 'data':data_buffer})
                        data_buffer = np.empty((1,2), dtype=float)
                        is_buffer_nonempty = False
                if fpga_finished or halt or (not main_ui_on_running):
                    session_STOP_ctrl.write(True)
                    QtCore.QThread.msleep(100)
                    session.close()
                    print('session closed')
                    if fpga_finished or halt:
                        # flush data in data buffer
                        if is_buffer_nonempty:
                            data_buffer = np.delete(data_buffer, obj=0, axis=0)
                            q_list[3].put({'sender':'FPGAManager', 'command':'save', 'data':data_buffer})
                            print('FGPA squeezed data into file')
                        # halt notifier
                        q_list[0].put({'sender':'FPGAManager', 'command':'done', 'data':None})
                        q_list[3].put({'sender':'FPGAManager', 'command':'halt', 'data':None})
                    break
                QtCore.QThread.msleep(50)
        if not main_ui_on_running:
            break

def plot_vec_crop(arr):
    buff_len = 10000
    if arr.shape[0] > buff_len:
        return arr[arr.shape[0]-buff_len:-1]
    else:
        return arr

def _FileManager(q_list):
    # File Manager uses queue_list[3]
    print('FileManager PID: ' + str(os.getpid()))
    main_ui_on_running = True
    while True:
        msg = q_list[3].get()
        if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
            main_ui_on_running = False
            break
        if (msg['sender'] != 'MyApp_Messenger') or (msg['sender'] == 'MyApp_Messenger' and msg['command'] != 'trigger'):
            continue
        with open(msg['data'], 'w', newline='') as csv_handle:
            csv_writer = csv.writer(csv_handle, delimiter=',')
            # data buffer
            data_buffer = np.empty((1,2), dtype=float)
            is_buffer_nonempty = False
            frame_timer_reset = 5.0
            frame_timer = frame_timer_reset
            while True:
                tic = time.time()
                if not q_list[3].empty():
                    msg = q_list[3].get()
                    if msg['sender'] == 'FPGAManager' and msg['command'] == 'save':
                        data_buffer = np.concatenate((data_buffer, msg['data']))
                        is_buffer_nonempty = True
                    if is_buffer_nonempty and frame_timer <= 0:
                        data_buffer = np.delete(data_buffer, obj=0, axis=0)
                        csv_writer.writerows(data_buffer)
                        data_buffer = np.empty((1,2), dtype=float)
                        frame_timer = frame_timer_reset
                        is_buffer_nonempty = False
                        print('data written')
                    if msg['sender'] == 'FPGAManager' and msg['command'] == 'halt':
                        if is_buffer_nonempty:
                            data_buffer = np.delete(data_buffer, obj=0, axis=0)
                            csv_writer.writerows(data_buffer)
                            print('data halt (FPGA) written')
                        is_buffer_nonempty = False
                        break
                    if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'halt':
                        if is_buffer_nonempty:
                            data_buffer = np.delete(data_buffer, obj=0, axis=0)
                            csv_writer.writerows(data_buffer)
                            print('data halt (Messenger) written')
                        is_buffer_nonempty = False
                        break
                    if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
                        main_ui_on_running = False
                        break
                QtCore.QThread.msleep(100)
                toc = time.time()
                frame_timer -= (toc - tic)
            csv_handle.close()
            print('file closed')
        if not main_ui_on_running:
            break

def _BackgroundManager(q_list):
    print('BackgroundManager PID: ' + str(os.getpid()))
    # BackgroundManager uses queue_list[1]
    # FPGA manager
    fpga_manager_p = Process(target=_FPGAManager, args=(q_list,))
    # file manager
    file_manager_p = Process(target=_FileManager, args=(q_list,))
    fpga_manager_p.start()
    file_manager_p.start()
    # event loop
    while True:
        msg = q_list[1].get()
        # message categorized
        if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'trigger':
            q_list[2].put(msg)
        if msg['sender'] == 'MyApp_Messenger' and msg['command'] == 'exit':
            break
    fpga_manager_p.join()
    file_manager_p.join()

# msg is dictionary {"sender": "", "command": "nop", "data": ...}

class MyApp_Messenger(QtCore.QThread):
    signal_update_graph = QtCore.pyqtSignal(object)
    signal_btn_unlock = QtCore.pyqtSignal(object)
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
        self.frame_timer_reset = 0.04
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
            if msg['sender'] == 'FPGAManager' and msg['command'] == 'draw' and frame_timer <= 0:
                self.signal_update_graph.emit(msg['data'])
                frame_timer = self.frame_timer_reset
            if msg['sender'] == 'FPGAManager' and msg['command'] == 'done':
                self.signal_btn_unlock.emit(msg)
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
    is_routine_running = False
    def __init__(self):
        super().__init__()
        print('MyApp PID: ' + str(os.getpid()))
        uic.loadUi('CV_Form.ui', self)
        # graph initialization
        self.graph_init()
        # message thread
        self.messenger = MyApp_Messenger()
        self.messenger.signal_update_graph.connect(self.update_graph)
        self.messenger.signal_btn_unlock.connect(self.btn_unlock)
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
        if self.is_routine_running:
            self.messenger.queue_list[2].put({'sender':'MyApp_Messenger', 'command':'halt', 'data':None})
            self.is_routine_running = False
            self.Btn_Run.setText('Run')
        else:
            # trigger File Manager
            self.messenger.queue_list[3].put({'sender':'MyApp_Messenger', 'command':'trigger', 'data':self.lineEdit_filepath.text()})
            # trigger FPGA Manager
            low_v = float(self.lineEdit_param_low_v.text())
            high_v = float(self.lineEdit_param_high_v.text())
            init_v = float(self.lineEdit_param_init_v.text())
            scan_vps = float(self.lineEdit_param_vps.text())
            cycles = int(self.lineEdit_param_cycles.text())
            segres = int(self.lineEdit_param_segres.text())
            pos_scan = self.checkBox_param_is_positive_scan.isChecked()
            param_dict = {'low_v':low_v, 'high_v':high_v, 'init_v':init_v, 'scan_vps':scan_vps, 'cycles':cycles, 'segment_res':segres, 'pos_scan':pos_scan}
            self.messenger.queue_list[2].put({'sender':'MyApp_Messenger', 'command':'trigger', 'data':param_dict})
            self.is_routine_running = True
            self.Btn_Run.setText('Stop')
    def btn_unlock(self):
        self.is_routine_running = False
        self.Btn_Run.setText('Run')
    def on_window_exit(self):
        self.messenger.queue_list[3].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.queue_list[2].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.queue_list[1].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.queue_list[0].put({'sender':'MyApp_Messenger', 'command':'exit', 'data':None})
        self.messenger.message_thread.quit()
        self.messenger.message_thread.wait()

