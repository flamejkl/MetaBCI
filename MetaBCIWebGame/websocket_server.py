# websocket_server.py
print("WS FILE =", __file__, flush=True)
print("🔥🔥🔥 修正版 websocket_server.py 已加载 🔥🔥🔥")
import sys
sys.path.insert(0, r"D:\pyproject\MetaBCI")

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
    VOTER_LOCK_DURATION, VOTER_THRESHOLD,
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

    # ========== 固定窗口评测任务 ==========
    async def _run_eval_fixed_window(self, expected_dir, websocket):
        """
        使用 Growing Window 解码器进行在线评测
        """
        log(f"[Eval] 开始执行 Growing Window 评测，期望方向={expected_dir}")

        # 初始化解码器（如果未初始化）
        decoder = self._init_gw_decoder()
        if decoder is None:
            error_msg = {
                "type": "eval_result",
                "decoded": None,
                "expected": expected_dir,
                "match": False,
                "timeout": True,
                "confidence": 0.0,
                "all_confidences": [0.0] * 4
            }
            await websocket.send(json.dumps(error_msg))
            return

        # 重置解码器状态（开始新试次）
        decoder.reset()

        # 最大等待时间（秒）
        max_wait = 3.5
        start_time = time.time()

        # 逐样本获取并喂入解码器
        while time.time() - start_time < max_wait:
            # 获取最新一个样本（假设 acq 有 get_latest_sample 方法）
            # 如果没有，可以使用 get_latest_samples(1) 取一个点
            try:
                if self.use_real and self.acq:
                    # 尝试获取最新样本（自行实现 get_latest_sample 或使用 chunk）
                    sample = await asyncio.to_thread(self.acq.get_latest_sample)
                    if sample is None:
                        await asyncio.sleep(0.001)
                        continue
                elif self.simulator is not None:
                    # 模拟器模式：直接模拟一个样本（这里需要根据模拟器调整）
                    # 简化处理：直接取模拟器生成的数据
                    sample = self.simulator.get_sample()  # 假设模拟器支持逐样本
                else:
                    # 无数据源
                    break
            except Exception as e:
                log(f"[Eval] 获取样本异常: {e}")
                await asyncio.sleep(0.001)
                continue

            # 喂入解码器
            decision, conf, current_time = decoder.feed(sample)

            if decision is not None:
                # 提前停止
                decoded_dir = ["up", "down", "left", "right"][decision]
                match = (decoded_dir == expected_dir)
                log(f"[Eval] 提前停止于 {current_time:.2f}s, 决策={decoded_dir}, 匹配={match}")
                result_msg = {
                    "type": "eval_result",
                    "decoded": decoded_dir,
                    "expected": expected_dir,
                    "match": match,
                    "timeout": False,
                    "confidence": conf,
                    "all_confidences": [0.0] * 4  # 可根据需要填充
                }
                await websocket.send(json.dumps(result_msg))
                return

            # 如果超过最大等待时间，退出循环
            if time.time() - start_time > max_wait:
                break

        # 超时处理：强制 2.0s 输出
        log("[Eval] 未提前停止，强制 2.0s 输出")
        # 获取最后 500 个点
        if self.use_real and self.acq:
            raw_window = await asyncio.to_thread(self.acq.get_latest_samples, 500)
            if raw_window is not None:
                window = raw_window[self.occipital_indices, :]
                window = window - np.mean(window, axis=1, keepdims=True)
                # 使用 2s 模型
                model = decoder.models[500]  # 假设 decoder.models 是字典
                X_input = window[np.newaxis, ...]
                scores = model.transform(X_input)[0]
                decision = np.argmax(scores)
                conf = np.max(scores)
                decoded_dir = ["up", "down", "left", "right"][decision]
                match = (decoded_dir == expected_dir)
                result_msg = {
                    "type": "eval_result",
                    "decoded": decoded_dir,
                    "expected": expected_dir,
                    "match": match,
                    "timeout": True,
                    "confidence": conf,
                    "all_confidences": [0.0] * 4
                }
                await websocket.send(json.dumps(result_msg))
                return

        # 所有尝试失败
        error_msg = {
            "type": "eval_result",
            "decoded": None,
            "expected": expected_dir,
            "match": False,
            "timeout": True,
            "confidence": 0.0,
            "all_confidences": [0.0] * 4
        }
        await websocket.send(json.dumps(error_msg))

    # ========== 实时解码启动（返回 bool） ==========
    async def _start_realtime_decoding(self, eval_mode=False, offline_mode=False):
        log(f"[START_DECODING] 进入函数, eval_mode={eval_mode}, offline_mode={offline_mode}")

        # ★★★ 关键修复：无论是否复用线程，都要先更新 eval_mode 状态 ★★★
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

        # 如果已有运行线程，直接返回 True（但上面已设置 eval_mode）
        if self.realtime_thread and self.realtime_thread.is_alive():
            log("[START_DECODING] 解码线程已运行，直接返回 True")
            return True

        # 以下为新建线程的逻辑
        self.realtime_stop_flag = threading.Event()

        # ===== 加载模型 =====
        self._load_model()
        if self.model is None:
            log("[START_DECODING] 模型加载失败，返回 False")
            self.eval_mode = False
            return False

        self.offline_mode = offline_mode

        # ===== 初始化 Growing Window 解码器（非评测模式） =====
        if not eval_mode:
            self._init_gw_decoder()
            if self.gw_decoder is None:
                log("[START_DECODING] Growing Window 解码器初始化失败")
                self.eval_mode = False
                return False
            else:
                self.gw_decoder.reset()
                log("[START_DECODING] Growing Window 解码器已就绪")

        # ===== 初始化离线数据生成器（仅离线模式） =====
        self.offline_gen = None
        if offline_mode:
            try:
                self.offline_gen = OfflineDataGenerator(
                    data_root=OFFLINE_DATA_ROOT,
                    window_samples=WINDOW_LEN_SAMPLES,
                    slide_step=25,
                    occipital_indices=[2, 3, 4, 5, 6, 7, 8, 9]
                )
                log("[START_DECODING] 离线数据生成器初始化成功")
            except Exception as e:
                log(f"[START_DECODING] 离线数据生成器初始化失败: {e}")
                self.eval_mode = False
                return False

        # ===== 连接真实 EEG 设备 =====
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

        # ===== 保留旧解码器（兼容性） =====
        if self.decoder is None:
            self.decoder = DynamicStoppingDecoder(model=self.model)

        # ===== 确保 eval_mode 在非评测模式下为 False =====
        if not eval_mode:
            self.eval_mode = False

        # ===== 定义实时解码循环（使用 Growing Window） =====
        def decode_loop():
            """实时解码循环（使用 Growing Window）"""
            # 获取解码器实例
            decoder = self.gw_decoder
            if decoder is None:
                log("❌ 解码器未初始化，无法启动实时循环")
                return

            # 重置解码器（连续控制开始时清空状态）
            decoder.reset()

            # 获取数据源
            acq = self.acq
            use_real = self.use_real

            # 用于存储最近一次决策时间（避免重复发送）
            last_decision_time = -1

            log("[实时] Growing Window 解码循环已启动")

            while not self.realtime_stop_flag.is_set():
                try:
                    # 1. 获取一个样本
                    if use_real and acq:
                        # 从真实设备获取最新一个样本
                        sample = acq.get_latest_sample()
                        if sample is None:
                            time.sleep(0.001)
                            continue
                    else:
                        # 无数据源，等待
                        time.sleep(0.001)
                        continue
                except Exception as e:
                    log(f"获取样本异常: {e}")
                    time.sleep(0.001)
                    continue

                # 2. 喂入解码器
                decision, conf, current_time = decoder.feed(sample)

                # 3. 如果有决策，且与上次不同（避免重复发送）
                if decision is not None and current_time != last_decision_time:
                    last_decision_time = current_time
                    # 将决策转换为方向字符串
                    command = ["up", "down", "left", "right"][decision]
                    log(f"[实时] 决策: {command}, 置信度: {conf:.3f}, 时间: {current_time:.2f}s")
                    # 广播给所有前端客户端
                    msg = {
                        "type": "realtime_command",
                        "command": command,
                        "confidence": conf,
                        "all_confidences": [0.0] * 4  # 可根据需要填充
                    }
                    # 使用 asyncio 广播
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast(json.dumps(msg)),
                        self.loop
                    )
            log("[实时] 解码循环已退出")

        # ===== 启动解码线程 =====
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
                        await websocket.send(json.dumps({"type": "offline_status", "status": "stopped"}))
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
                        # 旧演示，忽略
                        pass

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
                        force_log(f"eval_step 分支，eval_mode={self.eval_mode}")
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
                            self.eval_task = asyncio.create_task(self._run_eval_fixed_window(expected_dir, websocket))
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
        # 如果已有服务器，先关闭
        if self.server and self.loop:
            try:
                asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
            except:
                pass
            self.server = None
        # 如果线程仍在运行，等待结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            self.thread = None
        # 重新启动
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # 等待服务器真正启动
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