# websocket_server.py
print("WS FILE =", __file__, flush=True)
print("🔥🔥🔥 统一管道版 websocket_server.py 已加载 🔥🔥🔥")
import sys
import os
_METABCI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metabci')
if os.path.isdir(os.path.join(_METABCI_ROOT, 'brainda')):
    sys.path.insert(0, os.path.abspath(os.path.join(_METABCI_ROOT, '..')))

import asyncio
import json
import threading
import time
import websockets
import numpy as np
import joblib
import random
import traceback
from collections import deque
from typing import Optional, Callable, Awaitable, Tuple, Any
from config import (
    WS_HOST, WS_PORT, MODEL_PATH, NEURACLE_IP, NEURACLE_PORT,
    OFFLINE_DATA_ROOT, WINDOW_LEN_SAMPLES, ONLINE_SAMPLE_RATE,
    DEMO_DATA_ROOT
)
from data_acquisition import DataAcquisition

# ---- MetaBCI 框架集成 ----
from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder
from metabci.brainda.datasets import SelfSSVEP
from metabci.brainflow.logger import get_logger

_base_logger = get_logger("MetaBCIWebGame")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def force_log(msg):
    _base_logger.info(msg)
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================== 离线数据生成器（修改：存储文件名） ===========================
class OfflineDataGenerator:
    def __init__(self, data_root, window_samples=500, slide_step=25, occipital_indices=None, offset_only=True):
        self.data_root = data_root
        self.window_samples = window_samples
        self.slide_step = slide_step
        self.occipital_indices = occipital_indices if occipital_indices is not None else [2,3,4,5,6,7,8,9]
        self.offset_only = offset_only
        self.trials_by_label = {0: [], 1: [], 2: [], 3: []}
        self._load_all_trials()

    def _load_all_trials(self):
        # 使用 MetaBCI 数据集接口加载自采 SSVEP 数据
        ds = SelfSSVEP(
            data_root=self.data_root,
            occipital_indices=self.occipital_indices,
            offset_only=self.offset_only,
        )
        self.trials_by_label = ds.trials_by_label
        self.trials = []
        for label in range(4):
            self.trials.extend(self.trials_by_label[label])
        if not self.trials:
            raise RuntimeError("未加载到任何离线试次，请检查 data_self 目录")
        log(f"离线数据加载完成（MetaBCI SelfSSVEP），共 {len(self.trials)} 个试次")

    def get_full_trial_generator_by_label(self, label):
        trials = self.trials_by_label.get(label, [])
        if not trials:
            raise ValueError(f"标签 {label} 没有数据")
        n = len(trials)
        while True:
            idx = random.randint(0, n - 1)
            data, lbl, fname = trials[idx]
            yield data, lbl, fname   # 每次随机抽取


# =========================== 模拟数据生成器（未改动） ===========================
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


# =========================== 连续流解码引擎（修改：支持额外信息） ===========================
class ContinuousStreamingEngine:
    class State:
        IDLE = "IDLE"
        DEMO = "DEMO"
        EVAL = "EVAL"
        REALTIME = "REALTIME"

    def __init__(self, model, decoder, occipital_indices, sample_rate=250):
        self.model = model
        self.decoder = decoder
        self.occipital_indices = occipital_indices
        self.sample_rate = sample_rate

        self.state = self.State.IDLE
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # 数据源回调：返回 (data_chunk, is_new_trial, extra_info)
        self.data_source_callback: Optional[Callable[[], Tuple[Optional[np.ndarray], bool, dict]]] = None
        self.emit_callback: Optional[Callable[[dict], Awaitable[None]]] = None

        self.context = {
            "expected_dir": None,
            "msg_type": None,
            "trial_started": False,
        }
        self.current_extra = {}   # 当前试次的额外信息（如文件名）

        self.frame_count = 0
        self.decision_count = 0
        self.last_decision_time = 0.0

        log("[Engine] 连续流解码引擎初始化完成")

    async def _continuous_loop(self):
        self._running = True
        self._stop_event.clear()
        log("[Engine] 连续解码循环启动")

        while self._running:
            loop_start = time.perf_counter()

            if self.data_source_callback:
                result = self.data_source_callback()
                if result is not None:
                    data_chunk, is_new_trial, extra_info = result
                    if data_chunk is not None:
                        if is_new_trial and self.state in (self.State.DEMO, self.State.EVAL):
                            self.decoder.reset()
                            self.current_extra = extra_info or {}
                            log("[Engine] 新试次开始，decoder 已重置")
                        for sample in data_chunk:
                            if sample.shape[0] == 14:
                                sample = sample[self.occipital_indices]
                            decision, conf, current_time = self.decoder.feed(sample)
                            if decision is not None and self.state != self.State.IDLE:
                                await self._on_decision(decision, conf, current_time)

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 0.004 - elapsed)
            await asyncio.sleep(sleep_time)

        log("[Engine] 连续解码循环退出")

    async def _on_decision(self, decision, conf, current_time):
        self.decision_count += 1
        self.last_decision_time = current_time

        if self.state == self.State.REALTIME:
            command = ["up", "down", "left", "right"][decision]
            # 从decoder获取逐类分数（如果可用）
            scores = getattr(self.decoder, '_last_scores', None)
            if scores is not None:
                all_conf = [float(s) for s in scores]
            else:
                # 基于decision/conf估算分布
                all_conf = [0.0] * 4
                all_conf[decision] = conf
                rem = (1.0 - conf) / 3.0
                for i in range(4):
                    if i != decision:
                        all_conf[i] = max(0.0, rem)
            msg = {
                "type": "realtime_command",
                "command": command,
                "confidence": conf,
                "all_confidences": all_conf
            }
            if self.emit_callback:
                await self.emit_callback(msg)

        elif self.state in (self.State.DEMO, self.State.EVAL):
            expected = self.context.get("expected_dir")
            msg_type = self.context.get("msg_type", "eval_result")
            if expected is not None:
                decoded_dir = ["up", "down", "left", "right"][decision]
                match = (decoded_dir == expected)
                scores = getattr(self.decoder, '_last_scores', None)
                if scores is not None:
                    all_conf = [float(s) for s in scores]
                else:
                    all_conf = [0.0] * 4
                    all_conf[decision] = conf
                msg = {
                    "type": msg_type,
                    "decoded": decoded_dir,
                    "expected": expected,
                    "match": match,
                    "timeout": False,
                    "confidence": conf,
                    "all_confidences": all_conf
                }
                # 添加文件名（如果有）
                if 'filename' in self.current_extra:
                    msg['filename'] = self.current_extra['filename']
                if self.emit_callback:
                    await self.emit_callback(msg)
                self.context["expected_dir"] = None
                self.context["msg_type"] = None

        if self.state != self.State.REALTIME:
            self.decoder.reset()
        else:
            # REALTIME: slide window forward instead of full reset
            self.decoder.slide()

    async def start(self):
        if self._running:
            log("[Engine] 引擎已在运行")
            return
        self._loop_task = asyncio.create_task(self._continuous_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        self.state = self.State.IDLE
        self.context = {"expected_dir": None, "msg_type": None}
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self.decoder.reset()
        log("[Engine] 引擎已停止，状态 IDLE")

    def set_mode(self, state: str, expected_dir=None, msg_type=None):
        self.state = state
        self.context["expected_dir"] = expected_dir
        self.context["msg_type"] = msg_type
        log(f"[Engine] 模式切换为: {state}")

    # ===== 内置数据源回调（用于真实采集） =====
    def _next_acq_sample(self):
        """返回 (sample, False, {})，用于实时流"""
        sample = self.acq.get_latest_sample() if hasattr(self, 'acq') else None
        if sample is None:
            return None, False, {}
        return sample[np.newaxis, :], False, {}


# =========================== WebSocket 服务器主类（修改数据源回调） ===========================
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

        self.engine = None
        self.model = None
        self.gw_decoder = None
        self.acq = None
        self.offline_gen = None

        # 离线演示专用：当前试次生成器
        self._demo_gen = None

    # ========== 统一初始化 ==========
    def _load_model_and_engine(self):
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
                log(f"✅ 模型加载成功: {MODEL_PATH}")
            except Exception as e:
                log(f"❌ 模型加载失败: {e}")
                self.model = None
                return

        if self.engine is None:
            self._init_gw_decoder()
            if self.gw_decoder is None:
                log("❌ Growing Window 解码器初始化失败")
                return
            self.engine = ContinuousStreamingEngine(
                model=self.model,
                decoder=self.gw_decoder,
                occipital_indices=[2, 3, 4, 5, 6, 7, 8, 9]
            )
            self.engine.emit_callback = self._send_websocket
            # 注入 acq 引用供回调使用
            self.engine.acq = self.acq
            log("✅ 连续流引擎已创建")

    def _init_gw_decoder(self):
        if self.gw_decoder is None:
            try:
                self.gw_decoder = GrowingWindowDecoder(model_paths={
                    125: os.path.join(BASE_DIR, "model_125.pkl"),
                    250: os.path.join(BASE_DIR, "model_250.pkl"),
                    375: os.path.join(BASE_DIR, "model_375.pkl"),
                    500: os.path.join(BASE_DIR, "self_ssvep_model.pkl"),
                })
                log("✅ Growing Window 解码器已初始化")
            except Exception as e:
                log(f"❌ Growing Window 解码器初始化失败: {e}")
                self.gw_decoder = None
        return self.gw_decoder

    def _connect_acq(self):
        if self.acq is not None:
            return True
        try:
            self.acq = DataAcquisition(
                mode='real',
                neuracle_ip=NEURACLE_IP,
                neuracle_port=NEURACLE_PORT,
                srate=ONLINE_SAMPLE_RATE,
                num_chans=14
            )
            connected = self.acq.connect()
            if connected:
                self.acq.start_acquisition()
                self.acq.reset_buffer()
                log("✅ 真实 EEG 设备连接成功")
                # 更新引擎的acq引用
                if self.engine:
                    self.engine.acq = self.acq
                return True
            else:
                log("❌ 真实 EEG 设备连接失败")
                self.acq = None
                return False
        except Exception as e:
            log(f"❌ 真实 EEG 设备连接异常: {e}")
            self.acq = None
            return False

    # ========== WebSocket 发送回调 ==========
    async def _send_websocket(self, msg):
        await self._broadcast(json.dumps(msg))

    # ========== 统一停止（异步） ==========
    async def _stop_all(self):
        force_log("进入 _stop_all (async)")
        if self.engine:
            await self.engine.stop()
        if self.acq:
            self.acq.stop_acquisition()
            self.acq = None
        self.offline_gen = None
        self._demo_gen = None
        log("[STOP] 所有组件已停止")

    # ========== 离线演示数据源回调（修改：返回三元组） ==========
    def _get_demo_sample(self):
        """
        由引擎调用，返回 (data_chunk, is_new_trial, extra_info)
        每次 eval_step 设置 _demo_gen，本函数读取一个试次后立即清空，
        确保每个步骤只使用一个试次。
        """
        if self._demo_gen is not None:
            try:
                data, label, fname = next(self._demo_gen)  # 现在返回三个值
                self._demo_gen = None  # 只用一次
                log(f"[Demo] 获取试次，shape={data.shape}, label={label}, file={fname}")
                return data.T, True, {'filename': fname}   # 携带文件名
            except StopIteration:
                self._demo_gen = None
                return None, False, {}
        return None, False, {}

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

                    # ---------- 停止命令 ----------
                    if msg_type in ("stop_demo", "stop_eval", "stop_realtime"):
                        await self._stop_all()
                        await websocket.send(json.dumps({"type": "status", "status": "stopped"}))
                        continue

                    # ---------- 模式切换 ----------
                    if msg_type == "mode_switch":
                        new_mode = data.get("mode")
                        if new_mode in ["online", "offline"]:
                            self.mode = new_mode
                            force_log(f"模式切换为 {self.mode}")
                            if self.mode == "offline":
                                await self._stop_all()
                            await websocket.send(json.dumps({"type": "mode_switched", "mode": self.mode}))
                        continue

                    # ---------- 离线演示启动 ----------
                    if msg_type == "start_offline_sim":
                        await self._stop_all()
                        self._load_model_and_engine()
                        if self.engine is None:
                            await websocket.send(json.dumps({"type": "eval_error", "message": "引擎初始化失败"}))
                            continue

                        try:
                            self.offline_gen = OfflineDataGenerator(
                                data_root=DEMO_DATA_ROOT,
                                window_samples=WINDOW_LEN_SAMPLES,
                                slide_step=25,
                                occipital_indices=[2, 3, 4, 5, 6, 7, 8, 9],
                                offset_only=True
                            )
                        except Exception as e:
                            log(f"离线生成器初始化失败: {e}")
                            await websocket.send(json.dumps({"type": "eval_error", "message": str(e)}))
                            continue

                        self.engine.data_source_callback = self._get_demo_sample
                        self.engine.set_mode(ContinuousStreamingEngine.State.DEMO)
                        await self.engine.start()
                        await websocket.send(json.dumps({"type": "eval_started", "status": "ready"}))
                        continue

                    # ---------- 演示步骤（demo_step 兼容） ----------
                    if msg_type == "demo_step":
                        if self.engine and self.engine.state == ContinuousStreamingEngine.State.DEMO:
                            expected_dir = data.get("direction")
                            label = {"up":0, "down":1, "left":2, "right":3}[expected_dir]
                            self._demo_gen = self.offline_gen.get_full_trial_generator_by_label(label)
                            self.engine.decoder.reset()
                            self.engine.context["expected_dir"] = expected_dir
                            self.engine.context["msg_type"] = "demo_result"
                            log(f"[HANDLER] 设置 DEMO 期望: {expected_dir} (demo_step)")
                        else:
                            await websocket.send(json.dumps({"type": "error", "message": "演示未运行"}))
                        continue

                    # ---------- 评测步骤（同时支持离线演示） ----------
                    if msg_type == "eval_step":
                        if self.engine is None:
                            await websocket.send(json.dumps({"type": "error", "message": "引擎未初始化"}))
                            continue
                        if self.engine.state not in (ContinuousStreamingEngine.State.DEMO,
                                                     ContinuousStreamingEngine.State.EVAL):
                            await websocket.send(json.dumps({"type": "error", "message": "未处于演示或评测模式"}))
                            continue

                        expected_dir = data.get("direction")
                        if expected_dir not in ["up", "down", "left", "right"]:
                            await websocket.send(json.dumps({"type": "error", "message": "无效方向"}))
                            continue

                        self.engine.decoder.reset()
                        self.engine.context["expected_dir"] = expected_dir
                        # 根据状态决定 msg_type
                        if self.engine.state == ContinuousStreamingEngine.State.DEMO:
                            self.engine.context["msg_type"] = "demo_result"
                            # 设置离线数据源
                            if self.offline_gen is None:
                                await websocket.send(json.dumps({"type": "error", "message": "离线数据未加载"}))
                                continue
                            label = {"up": 0, "down": 1, "left": 2, "right": 3}[expected_dir]
                            self._demo_gen = self.offline_gen.get_full_trial_generator_by_label(label)
                            log(f"[HANDLER] 设置 DEMO 试次，方向: {expected_dir}")
                        else:
                            self.engine.context["msg_type"] = "eval_result"
                            log(f"[HANDLER] 设置 EVAL 期望: {expected_dir}")
                        continue

                    # ---------- 在线评测启动 ----------
                    if msg_type == "start_eval":
                        log("[HANDLER] 收到 start_eval")
                        if self.mode != "online":
                            await websocket.send(json.dumps({"type": "eval_error", "message": "请先切换到在线模式"}))
                            continue
                        await self._stop_all()
                        self._load_model_and_engine()
                        if self.engine is None:
                            await websocket.send(json.dumps({"type": "eval_error", "message": "引擎初始化失败"}))
                            continue

                        if not self._connect_acq():
                            await websocket.send(json.dumps({"type": "eval_error", "message": "EEG设备未连接"}))
                            continue

                        # 使用内置回调
                        self.engine.data_source_callback = self.engine._next_acq_sample
                        self.engine.set_mode(ContinuousStreamingEngine.State.EVAL)
                        await self.engine.start()
                        await websocket.send(json.dumps({"type": "eval_started", "status": "ready"}))
                        continue

                    # ---------- 实时脑控启动 ----------
                    if msg_type == "start_realtime":
                        log("[HANDLER] 收到 start_realtime")
                        if self.mode != "online":
                            await websocket.send(json.dumps({"error": "请先切换到在线模式"}))
                            continue
                        await self._stop_all()
                        self._load_model_and_engine()
                        if self.engine is None:
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "error", "message": "引擎初始化失败"}))
                            continue

                        if not self._connect_acq():
                            await websocket.send(json.dumps({"type": "realtime_status", "status": "error", "message": "EEG设备未连接"}))
                            continue

                        self.engine.data_source_callback = self.engine._next_acq_sample
                        self.engine.set_mode(ContinuousStreamingEngine.State.REALTIME)
                        await self.engine.start()
                        await websocket.send(json.dumps({"type": "realtime_status", "status": "started"}))
                        continue

                    # ---------- 其他消息 ----------
                    await self._broadcast(message)

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    force_log(f"[HANDLER] 消息处理异常: {e}")
                    traceback.print_exc()
                    try:
                        await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass
        finally:
            self.clients.remove(websocket)
            log("[HANDLER] 客户端断开")

    # ========== WebSocket 广播 ==========
    async def _broadcast(self, message):
        if not self.clients:
            return
        msg = json.dumps(message) if not isinstance(message, str) else message
        stale = []
        # Iterate over a snapshot to avoid "set changed size during iteration"
        for client in list(self.clients):
            try:
                await client.send(msg)
            except Exception:
                stale.append(client)
        # Clean up disconnected clients
        for client in stale:
            self.clients.discard(client)

    # ========== 服务器生命周期 ==========
    async def _start_server(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
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
        # 同步方式停止 - 使用 run_coroutine_threadsafe 并等待
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(self._stop_all(), self.loop)
            try:
                future.result(timeout=3)
            except Exception as e:
                log(f"停止时异常: {e}")
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)
        print("WebSocket 服务器已停止")


def get_websocket_server():
    return WebSocketServer()