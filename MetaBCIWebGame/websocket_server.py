# websocket_server.py
print("WS FILE =", __file__, flush=True)
print("🔥🔥🔥 修正版 websocket_server.py 已加载 🔥🔥🔥")
import sys
sys.path.insert(0, r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI")

import asyncio
import json
import threading
import time
import websockets
import numpy as np
import joblib
import os
import glob
import traceback
from collections import Counter
from config import (
    WS_HOST, WS_PORT, MODEL_PATH, NEURACLE_IP, NEURACLE_PORT,
    OFFLINE_DATA_ROOT, WINDOW_LEN_SAMPLES, ONLINE_SAMPLE_RATE,
    RAW_WINDOW_SAMPLES, VOTER_DECAY, VOTER_LOCK_FRAMES,
    VOTER_LOCK_DURATION, VOTER_THRESHOLD,DEMO_DATA_ROOT,
    FIXED_WINDOW_MODE
)
from online_decode import DynamicStoppingDecoder
from data_acquisition import DataAcquisition
from advanced_voter import AdvancedVoter
from growing_window_decoder import GrowingWindowDecoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def force_log(msg):
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()

# 强制刷新的日志函数
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================== 离线数据生成器 ===========================
class OfflineDataGenerator:
    def __init__(self, data_root, window_samples=500, slide_step=25, occipital_indices=None):
        self.data_root = data_root
        self.window_samples = window_samples
        self.slide_step = slide_step
        self.occipital_indices = occipital_indices if occipital_indices is not None else [2,3,4,5,6,7,8,9]
        self.trials_by_label = {0: [], 1: [], 2: [], 3: []}
        self._load_all_trials()

    def _load_all_trials(self):
        for label in range(4):
            folder = os.path.join(self.data_root, str(label + 1))
            if not os.path.isdir(folder):
                log(f"警告：目录 {folder} 不存在")
                continue
            files = glob.glob(os.path.join(folder, "*.npy"))
            for f in files:
                if "hw_trial_0000.npy" in f:
                    log(f"[跳过] 已知异常试次: {f}")
                    continue
                data = np.load(f)
                log(f"[加载] {f} shape={data.shape}")
                # 注意：这里已经提取枕区通道
                data = data[self.occipital_indices, :]
                self.trials_by_label[label].append((data, label))
        self.trials = []
        for label in range(4):
            self.trials.extend(self.trials_by_label[label])
        if not self.trials:
            raise RuntimeError("未加载到任何离线试次，请检查 data_self 目录")
        log(f"离线数据加载完成，共 {len(self.trials)} 个试次")

    def get_trial_generator_by_label(self, label):
        trials = self.trials_by_label.get(label, [])
        if not trials:
            raise ValueError(f"标签 {label} 没有数据")
        idx = 0
        while True:
            data, lbl = trials[idx % len(trials)]
            idx += 1
            n_samples = data.shape[1]
            for start in range(0, n_samples - self.window_samples + 1, self.slide_step):
                window = data[:, start:start+self.window_samples]
                yield window, lbl, (start + self.window_samples >= n_samples)
            if n_samples >= self.window_samples:
                yield data[:, -self.window_samples:], lbl, True

# =========================== 模拟数据生成器 ===========================
class SimulatedDataGenerator:
    def __init__(self, srate=250, n_channels=14, target_sequence=None,
                 switch_interval=2.0, window_len=WINDOW_LEN_SAMPLES):
        self.srate = srate
        self.n_channels = n_channels
        self.window_len = window_len
        self.target_sequence = target_sequence if target_sequence else ['up', 'down', 'left', 'right']
        self.switch_interval = switch_interval
        self.current_target_idx = 0
        self.last_switch_time = time.time()
        self.t = 0.0
        self.freq_map = {'up':8, 'down':10, 'left':12, 'right':15}

    def get_window(self):
        dt = 1.0 / self.srate
        now = time.time()
        if now - self.last_switch_time >= self.switch_interval:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.target_sequence)
            self.last_switch_time = now
            log(f"模拟目标切换为: {self.target_sequence[self.current_target_idx]}")
        target = self.target_sequence[self.current_target_idx]
        freq = self.freq_map[target]
        window = np.zeros((self.n_channels, self.window_len))
        for i in range(self.window_len):
            for ch in range(self.n_channels):
                window[ch, i] = 5e-5 * np.sin(2 * np.pi * freq * self.t) + 3e-5 * np.random.randn()
            self.t += dt
        return window

# =========================== WebSocket 服务器主类 ===========================
class WebSocketServer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.host = WS_HOST
        self.port = WS_PORT
        self.clients = set()
        self.server = None
        self.loop = None
        self.thread = None
        self._stop_event = threading.Event()
        self.mode = 'offline'
        self.realtime_thread = None
        self.realtime_stop_flag = None
        self.decoder = None
        self.model = None
        self.acq = None
        self.simulator = None
        self.voter = None

        self.eval_task = None
        self.gw_decoder = None

        # 评测模式相关
        self.eval_mode = False
        self.eval_active = False
        self.eval_lock = asyncio.Lock()
        self.eval_result_ready = threading.Event()
        self.eval_votes = []
        self.eval_start_time = 0.0
        self.eval_duration = 2.0
        self.eval_expected_dir = None
        self.eval_decoded = None

        # 用于固定窗口模式的状态
        self.use_real = False
        self.occipital_indices = [2, 3, 4, 5, 6, 7, 8, 9]
        self.offline_mode = False
        self.offline_gen = None

    # ========== 统一状态重置方法 ==========
    def reset_eval_state(self):
        """重置所有评测/演示相关状态（安全清理）"""
        self.eval_active = False
        self.eval_mode = False
        # 取消正在运行的 eval_task（如果有）
        if self.eval_task and not self.eval_task.done():
            self.eval_task.cancel()
        self.eval_task = None
        # 如果处于离线模式，停止实时解码线程
        if self.offline_mode:
            self._stop_realtime_decoding()
        log("[STATE] eval state reset")

    def _init_gw_decoder(self):
        """初始化 Growing Window 解码器（延迟加载）"""
        if self.gw_decoder is None:
            try:
                from growing_window_decoder import GrowingWindowDecoder
                self.gw_decoder = GrowingWindowDecoder()
                log("✅ Growing Window 解码器已初始化")
            except Exception as e:
                log(f"❌ Growing Window 解码器初始化失败: {e}")
                self.gw_decoder = None
        return self.gw_decoder

    def _load_model(self):
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
                log(f"✅ 模型加载成功: {MODEL_PATH}")
            except Exception as e:
                log(f"❌ 模型加载失败: {e}")
                self.model = None

    # ========== 评测任务（统一入口，支持 msg_type 参数） ==========
    async def _run_eval_fixed_window(self, expected_dir, websocket, msg_type="eval_result"):
        """
        执行 Growing Window 评测或演示。
        msg_type: "eval_result" 或 "demo_result"，决定返回消息类型。
        """
        self.eval_active = True
        try:
            log(f"[Eval] 开始评测，期望方向={expected_dir}, msg_type={msg_type}")

            # 初始化解码器
            decoder = self._init_gw_decoder()
            if decoder is None:
                await websocket.send(json.dumps({
                    "type": msg_type,
                    "decoded": None,
                    "expected": expected_dir,
                    "match": False,
                    "timeout": True,
                    "confidence": 0.0,
                    "all_confidences": [0.0] * 4
                }))
                return

            decoder.reset()
            max_wait = 3.5
            start_time = time.time()

            # ----- 离线模式处理 -----
            if self.offline_mode and self.offline_gen is not None:
                label_map = {"up": 0, "down": 1, "left": 2, "right": 3}
                label = label_map.get(expected_dir, 0)
                try:
                    gen = self.offline_gen.get_trial_generator_by_label(label)
                    data, true_label, _ = next(gen)
                    # data 已经是枕区通道 (8, 500)
                    for idx in range(data.shape[1]):
                        sample = data[:, idx]
                        decision, conf, current_time = decoder.feed(sample)
                        if decision is not None:
                            decoded_dir = ["up", "down", "left", "right"][decision]
                            match = (decoded_dir == expected_dir)
                            log(f"[Eval] 提前停止于 {current_time:.2f}s, 决策={decoded_dir}, 匹配={match}")
                            await websocket.send(json.dumps({
                                "type": msg_type,
                                "decoded": decoded_dir,
                                "expected": expected_dir,
                                "match": match,
                                "timeout": False,
                                "confidence": conf,
                                "all_confidences": [0.0] * 4
                            }))
                            return
                        if time.time() - start_time > max_wait:
                            break

                    # 未提前停止，强制 2.0s 输出
                    log("[Eval] 未提前停止，强制 2.0s 输出")
                    window_final = data[:, :500]
                    if window_final.shape[1] < 500:
                        window_final = data
                    # 注意：data 已是枕区，不再使用 self.occipital_indices 索引
                    window_final = window_final - np.mean(window_final, axis=1, keepdims=True)
                    model = decoder.models[500]
                    scores = model.transform(window_final[np.newaxis, ...])[0]
                    decision = np.argmax(scores)
                    conf = np.max(scores)
                    decoded_dir = ["up", "down", "left", "right"][decision]
                    match = (decoded_dir == expected_dir)
                    await websocket.send(json.dumps({
                        "type": msg_type,
                        "decoded": decoded_dir,
                        "expected": expected_dir,
                        "match": match,
                        "timeout": True,
                        "confidence": conf,
                        "all_confidences": [0.0] * 4
                    }))
                    return
                except StopIteration:
                    log("[Eval] 离线数据生成器无可用试次")
                except Exception as e:
                    log(f"[Eval] 离线数据异常: {e}")

            # ----- 在线模式（真实设备或模拟器）-----
            while time.time() - start_time < max_wait:
                try:
                    if self.use_real and self.acq:
                        sample = await asyncio.to_thread(self.acq.get_latest_sample)
                        if sample is None:
                            await asyncio.sleep(0.001)
                            continue
                    elif self.simulator is not None:
                        sample = self.simulator.get_sample()
                    else:
                        break
                except Exception as e:
                    log(f"[Eval] 获取样本异常: {e}")
                    await asyncio.sleep(0.001)
                    continue

                decision, conf, current_time = decoder.feed(sample)
                if decision is not None:
                    decoded_dir = ["up", "down", "left", "right"][decision]
                    match = (decoded_dir == expected_dir)
                    log(f"[Eval] 提前停止于 {current_time:.2f}s, 决策={decoded_dir}, 匹配={match}")
                    await websocket.send(json.dumps({
                        "type": msg_type,
                        "decoded": decoded_dir,
                        "expected": expected_dir,
                        "match": match,
                        "timeout": False,
                        "confidence": conf,
                        "all_confidences": [0.0] * 4
                    }))
                    return
                if time.time() - start_time > max_wait:
                    break

            # 超时处理
            log("[Eval] 超时，强制输出")
            if self.use_real and self.acq:
                raw_window = await asyncio.to_thread(self.acq.get_latest_samples, 500)
                if raw_window is not None:
                    window = raw_window[self.occipital_indices, :]
                    window = window - np.mean(window, axis=1, keepdims=True)
                    model = decoder.models[500]
                    scores = model.transform(window[np.newaxis, ...])[0]
                    decision = np.argmax(scores)
                    conf = np.max(scores)
                    decoded_dir = ["up", "down", "left", "right"][decision]
                    match = (decoded_dir == expected_dir)
                    await websocket.send(json.dumps({
                        "type": msg_type,
                        "decoded": decoded_dir,
                        "expected": expected_dir,
                        "match": match,
                        "timeout": True,
                        "confidence": conf,
                        "all_confidences": [0.0] * 4
                    }))
                    return

            # 兜底
            await websocket.send(json.dumps({
                "type": msg_type,
                "decoded": None,
                "expected": expected_dir,
                "match": False,
                "timeout": True,
                "confidence": 0.0,
                "all_confidences": [0.0] * 4
            }))

        except Exception as e:
            log(f"[Eval] ❌ 异常: {e}")
            try:
                await websocket.send(json.dumps({
                    "type": msg_type,
                    "decoded": None,
                    "expected": expected_dir,
                    "match": False,
                    "timeout": True,
                    "confidence": 0.0,
                    "all_confidences": [0.0] * 4
                }))
            except:
                pass
        finally:
            self.eval_active = False
            self.eval_task = None
            log("[Eval] 状态已重置")

    # ========== 实时解码启动（返回 bool） ==========
    async def _start_realtime_decoding(self, eval_mode=False, offline_mode=False):
        log(f"[START_DECODING] 进入函数, eval_mode={eval_mode}, offline_mode={offline_mode}")

        if eval_mode:
            self.eval_mode = True
            self.eval_active = False
            self.eval_votes = []
            self.eval_result_ready.clear()
            self.eval_decoded = None
            force_log("设置 eval_mode = True，调用栈:")
            traceback.print_stack(file=sys.stderr)
        else:
            self.eval_mode = False

        if self.realtime_thread and self.realtime_thread.is_alive():
            log("[START_DECODING] 解码线程已运行，直接返回 True")
            return True

        self.realtime_stop_flag = threading.Event()
        self._load_model()
        if self.model is None:
            log("[START_DECODING] 模型加载失败，返回 False")
            self.eval_mode = False
            return False

        self.offline_mode = offline_mode

        if not eval_mode:
            self._init_gw_decoder()
            if self.gw_decoder is None:
                log("[START_DECODING] Growing Window 解码器初始化失败")
                self.eval_mode = False
                return False
            else:
                self.gw_decoder.reset()
                log("[START_DECODING] Growing Window 解码器已就绪")

        self.offline_gen = None
        if offline_mode:
            try:
                self.offline_gen = OfflineDataGenerator(
                    data_root=DEMO_DATA_ROOT,
                    window_samples=WINDOW_LEN_SAMPLES,
                    slide_step=25,
                    occipital_indices=[2, 3, 4, 5, 6, 7, 8, 9]
                )
                log("[START_DECODING] 离线数据生成器初始化成功")
            except Exception as e:
                log(f"[START_DECODING] 离线数据生成器初始化失败: {e}")
                self.eval_mode = False
                return False

        use_real = False
        if not offline_mode:
            try:
                self.acq = DataAcquisition(
                    mode='real',
                    neuracle_ip=NEURACLE_IP,
                    neuracle_port=NEURACLE_PORT,
                    srate=ONLINE_SAMPLE_RATE,
                    num_chans=14
                )
                log("[START_DECODING] DataAcquisition 实例创建完毕，尝试连接...")
                connected = await asyncio.to_thread(self.acq.connect)
                if connected:
                    await asyncio.to_thread(self.acq.start_acquisition)
                    await asyncio.to_thread(self.acq.reset_buffer)
                    use_real = True
                    log("✅ 真实 EEG 设备连接成功，开始采集")
                else:
                    log("❌ 真实 EEG 设备连接失败")
                    self.acq = None
            except Exception as e:
                log(f"❌ 真实 EEG 设备连接异常: {e}")
                self.acq = None
            if not use_real:
                log("[START_DECODING] 真实设备不可用，返回 False")
                self.eval_mode = False
                return False
        else:
            self.acq = None

        self.use_real = use_real
        self.occipital_indices = [2, 3, 4, 5, 6, 7, 8, 9]

        if not eval_mode:
            self.eval_mode = False

        def decode_loop():
            decoder = self.gw_decoder
            if decoder is None:
                log("❌ 解码器未初始化，无法启动实时循环")
                return
            decoder.reset()
            acq = self.acq
            use_real = self.use_real
            last_decision_time = -1
            log("[实时] Growing Window 解码循环已启动")
            while not self.realtime_stop_flag.is_set():
                try:
                    if use_real and acq:
                        sample = acq.get_latest_sample()
                        if sample is None:
                            time.sleep(0.001)
                            continue
                    else:
                        time.sleep(0.001)
                        continue
                except Exception as e:
                    log(f"获取样本异常: {e}")
                    time.sleep(0.001)
                    continue

                decision, conf, current_time = decoder.feed(sample)
                if decision is not None and current_time != last_decision_time:
                    last_decision_time = current_time
                    command = ["up", "down", "left", "right"][decision]
                    log(f"[实时] 决策: {command}, 置信度: {conf:.3f}, 时间: {current_time:.2f}s")
                    msg = {
                        "type": "realtime_command",
                        "command": command,
                        "confidence": conf,
                        "all_confidences": [0.0] * 4
                    }
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast(json.dumps(msg)),
                        self.loop
                    )
            log("[实时] 解码循环已退出")

        self.realtime_thread = threading.Thread(target=decode_loop, daemon=True)
        self.realtime_thread.start()
        log("[START_DECODING] 实时解码线程已启动，返回 True")
        return True

    # ========== 停止解码 ==========
    def _stop_realtime_decoding(self):
        force_log("进入 _stop_realtime_decoding，调用栈:")
        traceback.print_stack(file=sys.stderr)
        log("[STOP_DECODING] 开始停止解码")
        if self.realtime_stop_flag:
            self.realtime_stop_flag.set()
        if self.realtime_thread:
            self.realtime_thread.join(timeout=2)
            self.realtime_thread = None
        if self.acq:
            self.acq.stop_acquisition()
            self.acq = None
        self.decoder = None
        self.simulator = None
        self.voter = None
        self.eval_mode = False
        force_log("设置 eval_mode = False (在 _stop_realtime_decoding 中)")
        self.eval_active = False
        self.eval_decoded = None
        log("实时解码已停止")

    # ========== 评测步骤控制（滑动窗口模式用） ==========
    def start_eval_step(self, expected_dir):
        if not self.eval_mode or self.realtime_thread is None or not self.realtime_thread.is_alive():
            return False
        if self.eval_active:
            return False
        with self.eval_lock:
            self.eval_votes = []
        if self.voter:
            self.voter.reset()
        self.eval_active = True
        self.eval_start_time = time.time()
        self.eval_expected_dir = expected_dir
        self.eval_decoded = None
        self.eval_result_ready.clear()
        log(f"评测步骤开始，期望方向: {expected_dir}")
        return True

    def wait_eval_result(self, timeout=3.0):
        if self.eval_result_ready.wait(timeout):
            return self.eval_decoded, self.eval_expected_dir
        else:
            self.eval_active = False
            return None, self.eval_expected_dir

    # ========== WebSocket 消息处理 ==========
    async def _handler(self, websocket):
        print("===== NEW HANDLER CALLED =====", flush=True)
        self.clients.add(websocket)
        log("[HANDLER] 新客户端连接")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    log(f"[HANDLER] 收到消息类型: {msg_type}")

                    if msg_type == "stop_demo":
                        # 重置所有评测/演示状态
                        self.reset_eval_state()
                        await websocket.send(json.dumps({"type": "offline_status", "status": "stopped"}))
                        log("[HANDLER] 已停止演示")
                        continue

                    if msg_type == "mode_switch":
                        new_mode = data.get("mode")
                        if new_mode in ["online", "offline"]:
                            self.mode = new_mode
                            force_log(f"模式切换为 {self.mode}")
                            if self.mode == "offline" and self.realtime_thread and self.realtime_thread.is_alive():
                                force_log("切换到 offline，即将调用 _stop_realtime_decoding")
                                self._stop_realtime_decoding()
                            await websocket.send(json.dumps({"type": "mode_switched", "mode": self.mode}))
                        continue

                    if msg_type == "demo_step":
                        # 处理离线演示步骤（独立于 eval）
                        expected_dir = data.get("direction")
                        log(f"[HANDLER] 收到 demo_step, expected={expected_dir}")
                        # 使用 demo_result 作为消息类型，不污染 eval_mode
                        self.eval_task = asyncio.create_task(
                            self._run_eval_fixed_window(expected_dir, websocket, msg_type="demo_result")
                        )
                        continue

                    if msg_type == "start_offline_sim":
                        log("[HANDLER] 收到 start_offline_sim")
                        if self.realtime_thread and self.realtime_thread.is_alive():
                            self._stop_realtime_decoding()
                        success = await self._start_realtime_decoding(eval_mode=True, offline_mode=True)
                        if success:
                            await asyncio.sleep(0.3)
                            await websocket.send(json.dumps({"type": "eval_started", "status": "ready"}))
                            log("[HANDLER] 发送 eval_started (离线)")
                        else:
                            await websocket.send(json.dumps({"type": "eval_error", "message": "离线数据加载失败"}))
                            log("[HANDLER] 发送 eval_error (离线)")
                        continue

                    if msg_type == "start_realtime":
                        log("[HANDLER] 收到 start_realtime")
                        if self.mode != "online":
                            await websocket.send(json.dumps({"error": "请先切换到在线模式"}))
                            continue
                        if self.realtime_thread and self.realtime_thread.is_alive():
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "already_running"}))
                            continue
                        success = await self._start_realtime_decoding(eval_mode=False, offline_mode=False)
                        if success:
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "started"}))
                            log("[HANDLER] 发送 realtime_status started")
                        else:
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "error", "message": "设备连接失败"}))
                            log("[HANDLER] 发送 realtime_status error")
                        continue

                    if msg_type == "stop_realtime":
                        log("[HANDLER] 收到 stop_realtime")
                        if self.realtime_thread and self.realtime_thread.is_alive():
                            self._stop_realtime_decoding()
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "stopped"}))
                        else:
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "not_running"}))
                        continue

                    if msg_type == "start_eval":
                        force_log(f"start_eval 分支，当前 eval_mode={self.eval_mode}")
                        log("[HANDLER] 收到 start_eval")
                        if self.mode != "online":
                            await websocket.send(
                                json.dumps({"type": "eval_error", "message": "请先切换到在线演示模式"}))
                            continue
                        if self.realtime_thread and self.realtime_thread.is_alive():
                            self._stop_realtime_decoding()
                        success = await self._start_realtime_decoding(eval_mode=True, offline_mode=False)
                        force_log(f"_start_realtime_decoding 返回 success={success}, eval_mode={self.eval_mode}")
                        if success:
                            await websocket.send(json.dumps({"type": "eval_started", "status": "ready"}))
                            log("[HANDLER] 发送 eval_started (在线)")
                        else:
                            await websocket.send(json.dumps({"type": "eval_error", "message": "EEG设备未连接，请检查设备后重试"}))
                            log("[HANDLER] 发送 eval_error (在线)")
                        continue

                    if msg_type == "eval_step":
                        # ---- 保险门 ----
                        if self.eval_task and self.eval_task.done():
                            self.eval_active = False
                            self.eval_task = None

                        if self.eval_active:
                            await websocket.send(json.dumps({"type": "eval_processing", "message": "正在处理中"}))
                            continue

                        expected_dir = data.get("direction")
                        log(f"[HANDLER] 收到 eval_step, expected={expected_dir}, eval_mode={self.eval_mode}, eval_active={self.eval_active}")

                        if not self.eval_mode:
                            force_log("即将返回 '评测模式未激活'，调用栈:")
                            traceback.print_stack(file=sys.stderr)
                            await websocket.send(json.dumps({"type": "eval_error", "message": "评测模式未激活"}))
                            log("[HANDLER] 发送 eval_error: 评测模式未激活")
                            continue

                        async with self.eval_lock:
                            if self.eval_active:
                                await websocket.send(json.dumps({"type": "eval_processing", "message": "正在处理中"}))
                                log("[HANDLER] 发送 eval_processing")
                                continue

                            self.eval_active = True
                            self.eval_expected_dir = expected_dir

                            if self.eval_task and not self.eval_task.done():
                                self.eval_task.cancel()
                            self.eval_task = asyncio.create_task(
                                self._run_eval_fixed_window(expected_dir, websocket, msg_type="eval_result")
                            )
                            log(f"[HANDLER] 已创建 eval_task，expected={expected_dir}")
                        continue

                    if msg_type == "stop_eval":
                        log("[HANDLER] 收到 stop_eval")
                        self.eval_mode = False
                        self.eval_active = False
                        self._stop_realtime_decoding()
                        await websocket.send(json.dumps({"type": "eval_stopped"}))
                        continue

                    # 其他消息广播
                    await self._broadcast(message)

                except json.JSONDecodeError:
                    pass
        finally:
            self.clients.remove(websocket)
            log("[HANDLER] 客户端断开")

    async def _broadcast(self, message):
        if not self.clients:
            return
        msg = json.dumps(message) if not isinstance(message, str) else message
        tasks = [asyncio.create_task(client.send(msg)) for client in self.clients]
        if tasks:
            await asyncio.wait(tasks)

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, self.host, self.port, reuse_address=True)
        await self.server.wait_closed()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start_server())
        except RuntimeError as e:
            log(f"事件循环运行异常: {e}")
        finally:
            self.loop.run_forever()

    def start(self):
        if self.server and self.loop:
            try:
                asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
            except:
                pass
            self.server = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            self.thread = None
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        while self.server is None:
            time.sleep(0.1)
        print(f"WebSocket 服务器已启动，监听 {self.host}:{self.port}")

    def stop(self):
        self._stop_event.set()
        self._stop_realtime_decoding()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)
        print("WebSocket 服务器已停止")

def get_websocket_server():
    return WebSocketServer()