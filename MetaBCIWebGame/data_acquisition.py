# data_acquisition.py
import time
import numpy as np
import threading
from neuracle_api import DataServerThread

class DataAcquisition:
    def __init__(self, mode='simulate', save_path=None,
                 neuracle_ip='127.0.0.1', neuracle_port=8712,
                 srate=250, num_chans=14):
        self.mode = mode
        self.save_path = save_path
        self.neuracle_ip = neuracle_ip
        self.neuracle_port = neuracle_port
        self.srate = srate
        # 目标通道列表（枕顶区，按训练所需顺序）
        self.target_channels = [
            'Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO5', 'PO6', 'POz',
            'P3', 'P4', 'P7', 'P8'
        ]
        self.num_chans = len(self.target_channels)
        self.channel_indices = None
        self.server = None
        self._stop_flag = threading.Event()
        self._acquisition_thread = None
        self.eeg_buffer = []          # 只存目标通道
        self.eeg_buffer_full = []     # 存所有通道（含Trigger）
        self._lock = threading.Lock()
        self._lock_full = threading.Lock()

    def connect(self):
        if self.mode == 'real':
            try:
                self.server = DataServerThread(sample_rate=self.srate, t_buffer=60)
                self.server.connect(hostname=self.neuracle_ip, port=self.neuracle_port)
                while not self.server.isReady():
                    time.sleep(0.01)
                all_ch = self.server.channelNames
                print(f"[Neuracle API] 设备总通道数: {len(all_ch)}")
                self.channel_indices = []
                for ch in self.target_channels:
                    if ch in all_ch:
                        self.channel_indices.append(all_ch.index(ch))
                    else:
                        raise ValueError(f"目标通道 {ch} 未在设备中找到！")
                matched_names = [all_ch[i] for i in self.channel_indices]
                print(f"[Neuracle API] 目标通道索引: {self.channel_indices}")
                print(f"[Neuracle API] 实际提取通道: {matched_names}")
                self.server.start()
                print(f"[Neuracle API] 已连接 {self.neuracle_ip}:{self.neuracle_port}")
                return True
            except Exception as e:
                print(f"[Neuracle API] 连接失败: {e}")
                self.server = None
                return False
        else:
            print("[Simulation] 模拟模式，无硬件连接")
            return True

    def _data_collection_loop(self):
        self.eeg_buffer = []
        self.eeg_buffer_full = []
        if self.mode == 'real' and self.server is None:
            print("错误：真实模式下未初始化 server")
            return

        while not self._stop_flag.is_set():
            if self.mode == 'real' and self.server is not None:
                try:
                    new_data = self.server.GetBufferUpdate()   # (channels, samples)
                    if new_data is not None and new_data.size > 0:
                        # 保存完整数据（所有通道）
                        with self._lock_full:
                            self.eeg_buffer_full.extend(new_data.T.tolist())
                        # 提取目标通道
                        eeg_chunk = new_data[self.channel_indices, :]
                        samples_list = eeg_chunk.T.tolist()
                        with self._lock:
                            self.eeg_buffer.extend(samples_list)
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    print(f"[Neuracle API] 采集异常: {e}")
                    time.sleep(0.01)
            else:
                # 模拟模式
                sample = np.random.randn(self.num_chans) * 1e-5
                with self._lock:
                    self.eeg_buffer.append(sample.tolist())
                time.sleep(1.0 / self.srate)

        if self.save_path and self.eeg_buffer:
            full_data = np.array(self.eeg_buffer).T
            import scipy.io as sio
            sio.savemat(self.save_path, {'eeg_data': full_data})
            print(f"数据已保存至 {self.save_path}")

    def start_acquisition(self):
        if self.mode == 'real' and self.server is None:
            if not self.connect():
                print("真实模式连接失败，无法启动采集")
                return
        self._stop_flag.clear()
        self._acquisition_thread = threading.Thread(target=self._data_collection_loop)
        self._acquisition_thread.start()
        print("EEG 采集线程已启动")

    def stop_acquisition(self):
        self._stop_flag.set()
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=5)
            self._acquisition_thread = None
        if self.server:
            self.server.stop()
            self.server = None
        print("EEG 采集已停止")

    def reset_buffer(self):
        with self._lock:
            self.eeg_buffer = []
        with self._lock_full:
            self.eeg_buffer_full = []

    def get_sample_count(self):
        with self._lock:
            return len(self.eeg_buffer)

    def get_latest_samples(self, n_samples):
        with self._lock:
            if len(self.eeg_buffer) < n_samples:
                return None
            data = np.array(self.eeg_buffer[-n_samples:], dtype=np.float64).T
            return data

    def get_all_data(self):
        with self._lock:
            if not self.eeg_buffer:
                return np.array([])
            return np.array(self.eeg_buffer, dtype=np.float64).T

    def get_latest_sample(self):
        """获取最新一个采样点（所有通道）"""
        with self._lock:
            if not self.eeg_buffer:
                return None
            # 返回最后一个采样点（列向量）
            return np.array(self.eeg_buffer[-1])

    # 新增：获取完整数据（含Trigger）
    def get_all_data_with_trigger(self):
        with self._lock_full:
            if not self.eeg_buffer_full:
                return np.array([])
            return np.array(self.eeg_buffer_full, dtype=np.float64).T