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
    DEMO_DATA_ROOT, GW_MODEL_PATHS
)
from data_acquisition import DataAcquisition

# ---- MetaBCI 框架集成 ----
from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder
from metabci.brainda.datasets import SelfSSVEP
from metabci.brainflow.online import ContinuousStreamingEngine
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


# =========================== 连续流解码引擎（从 brainflow 导入） ===========================
# ContinuousStreamingEngine 已集成至 metabci/brainflow/online.py
# 应用层仅保留数据源回调以注入 WebSocketServer 的 acq 引用


def _make_next_acq_sample(acq):
    """构建实时采集数据源回调（闭包捕获 acq 引用）。

    每次调用返回自上次读取以来 eeg_buffer 中新增的所有采样点，
    作为 batch 一次性喂给解码器，避免逐点重复读取导致缓冲区充满重复值。
    """
    read_cursor = [0]

    def _next():
        if acq is None:
            return None, False, {}
        with acq._lock:
            total = len(acq.eeg_buffer)
            if total <= read_cursor[0]:
                return None, False, {}
            new = acq.eeg_buffer[read_cursor[0]:total]
            read_cursor[0] = total
        if not new:
            return None, False, {}
        data = np.array(new, dtype=np.float64)          # (n_new, n_channels)
        return data, False, {}

    def _skip_stale():
        """跳到eeg_buffer末尾，丢弃旧数据（与decoder.reset()配套使用）。"""
        if acq is None:
            return
        with acq._lock:
            read_cursor[0] = len(acq.eeg_buffer)

    return _next, _skip_stale


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
        self.stim_clients = set()       # PsychoPy 刺激窗口客户端
        self.server = None
        self.loop = None
        self.thread = None
        self._stop_event = threading.Event()
        self.mode = 'offline'

        self.engine = None
        self.model = None
        # 0.5s 诊断统计
        self._diag_correct = 0
        self._diag_total = 0
        self._diag_trigger_miss = 0
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
            log("✅ 连续流引擎已创建")

    def _init_gw_decoder(self):
        if self.gw_decoder is None:
            try:
                self.gw_decoder = GrowingWindowDecoder(model_paths=GW_MODEL_PATHS, enable_online_norm=True)
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
            self.engine = None       # 必须置空，否则下次 _load_model_and_engine 复用已停止的引擎
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

                    # ---------- PsychoPy 刺激窗口注册 ----------
                    if msg_type == "stim_register":
                        self.stim_clients.add(websocket)
                        log("[HANDLER] PsychoPy 刺激窗口已注册")
                        continue

                    # ---------- 停止命令 ----------
                    if msg_type in ("stop_demo", "stop_eval", "stop_realtime"):
                        await self._stop_all()
                        await self._broadcast_stim({"type": "stim_stop"})
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
                            self.engine.request_reset()
                            if hasattr(self, '_skip_stale') and self._skip_stale:
                                self._skip_stale()
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
                                                     ContinuousStreamingEngine.State.EVAL,
                                                     ContinuousStreamingEngine.State.IDLE):
                            await websocket.send(json.dumps({"type": "error", "message": "未处于演示或评测模式"}))
                            continue

                        expected_dir = data.get("direction")
                        if expected_dir not in ["up", "down", "left", "right"]:
                            await websocket.send(json.dumps({"type": "error", "message": "无效方向"}))
                            continue

                        # ---- PsychoPy 刺激流程: Trigger精确提取 + 直接模型预测 ----
                        if self.stim_clients:
                            await self._broadcast_stim({"type": "stim_target", "direction": expected_dir})
                            await self._broadcast_stim({"type": "stim_phase", "phase": "index"})
                            await asyncio.sleep(0.5)  # 提示0.5s即可
                            if hasattr(self, '_skip_stale') and self._skip_stale:
                                self._skip_stale()
                            await self._broadcast_stim({"type": "stim_phase", "phase": "stimulus", "direction": expected_dir})
                            # ====== 固定2.0s窗口(W=500)在线解码 ======
                            import time as _time
                            W = 500
                            model = self.gw_decoder.models[W]
                            occ_idx = [2, 3, 4, 5, 6, 7, 8, 9]

                            # 1) 动态检测Trigger（同collect_step逻辑）
                            start = self.acq.get_sample_count()
                            deadline = _time.time() + 5.0
                            full = None
                            trigger_ch = None
                            onset = 0
                            while _time.time() < deadline:
                                await asyncio.sleep(0.05)
                                end = self.acq.get_sample_count()
                                if end - start < 200:
                                    continue
                                with self.acq._lock_full:
                                    full = np.array(self.acq.eeg_buffer_full[start:end], dtype=np.float64).T
                                trigger_ch = full[-1, :]
                                for i in range(len(trigger_ch) - 1):
                                    if trigger_ch[i] < 0.5 and trigger_ch[i+1] >= 0.5:
                                        onset = i + 1
                                        break
                                if onset > 0 and onset + W <= full.shape[1]:
                                    break

                            t_uniq = np.unique(trigger_ch[:100]) if trigger_ch is not None else []
                            if onset == 0 or full is None or onset + W > full.shape[1]:
                                onset = 0
                                self._diag_trigger_miss += 1
                                total = full.shape[1] if full is not None else 0
                                log(f"[DIAG] ⚠️ 未检测到Trigger跳变! "
                                    f"前100点唯一值={t_uniq.tolist()}, 总长={total}")
                                if full is None:
                                    await websocket.send(json.dumps({"type": "eval_result", "decoded": "error",
                                        "expected": expected_dir, "match": False, "confidence": 0,
                                        "decision_time": 2.0, "early": False, "window_ms": 2000}))
                                    continue
                            else:
                                log(f"[DIAG] ✓ Trigger onset={onset}, "
                                    f"前100点唯一值={t_uniq.tolist()}, "
                                    f"跳变: {trigger_ch[onset-1]:.1f}→{trigger_ch[onset]:.1f}")

                            # 2) 提取数据 + 预处理（对齐训练）
                            raw = full[self.acq.channel_indices, onset:onset + W]
                            trial = raw.astype(np.float64)
                            trial = trial[occ_idx, :]                    # 枕区8通道
                            trial = trial - np.mean(trial, axis=1, keepdims=True)

                            # 3) 数据质量日志
                            if self._diag_total % 10 == 1:
                                ch_std = np.std(trial, axis=1)
                                log(f"[DIAG] EEG质量: 幅值范围=[{trial.min():.1f},{trial.max():.1f}], "
                                    f"通道std={[f'{s:.1f}' for s in ch_std]}")

                            # 4) 预测 + 真实置信度
                            scores = model.transform(trial[np.newaxis, ...])[0]
                            decision = int(np.argmax(scores))
                            conf = float(np.max(scores))
                            # margin: top1 vs top2 差距
                            top2 = np.partition(scores, -2)[-2:]
                            margin = float(top2.max() - top2.min())

                            dec_t = W / 250.0
                            decoded_dir = ["up", "down", "left", "right"][decision]
                            match = (decoded_dir == expected_dir)

                            # 5) 累计统计
                            self._diag_total += 1
                            if match:
                                self._diag_correct += 1
                            diag_acc = self._diag_correct / self._diag_total * 100
                            miss_rate = self._diag_trigger_miss / self._diag_total * 100

                            log(f"[DIAG] #{self._diag_total} 目标={expected_dir} "
                                f"解码={decoded_dir} {'✓' if match else '✗'} "
                                f"conf={conf:.3f} margin={margin:.3f} "
                                f"累计准确率={diag_acc:.1f}% ({self._diag_correct}/{self._diag_total}) "
                                f"Trigger丢失率={miss_rate:.1f}%")

                            await websocket.send(json.dumps({
                                "type": "eval_result", "decoded": decoded_dir,
                                "expected": expected_dir, "match": match,
                                "confidence": round(float(conf), 3),
                                "margin": round(margin, 3),
                                "decision_time": round(float(dec_t), 3),
                                "early": float(dec_t) < 2.0,
                                "window_ms": int(float(dec_t) * 1000),
                                "diag_acc_pct": round(diag_acc, 1),
                                "diag_total": self._diag_total,
                            }))
                            continue  # 跳过引擎流程
                        else:
                            self.engine.request_reset()
                            if hasattr(self, '_skip_stale') and self._skip_stale:
                                self._skip_stale()
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
                        # 重置0.5s诊断统计
                        self._diag_correct = 0
                        self._diag_total = 0
                        self._diag_trigger_miss = 0
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
                        next_fn, skip_stale = _make_next_acq_sample(self.acq)
                        self._skip_stale = skip_stale
                        self.engine.data_source_callback = next_fn
                        # PsychoPy模式下不启动引擎(避免与eval handler线程竞争decoder)
                        if not self.stim_clients:
                            self.engine.set_mode(ContinuousStreamingEngine.State.EVAL)
                            await self.engine.start()
                        else:
                            self.engine.set_mode(ContinuousStreamingEngine.State.IDLE)
                        await websocket.send(json.dumps({"type": "eval_started", "status": "ready"}))
                        await self._broadcast_stim({"type": "stim_start"})
                        continue

                    # ---------- 浏览器数据采集 ----------
                    if msg_type == "start_collect":
                        log("[HANDLER] 收到 start_collect")
                        if self.mode != "online":
                            await websocket.send(json.dumps({"type": "collect_error", "message": "请先切换到在线模式"}))
                            continue
                        await self._stop_all()
                        if not self._connect_acq():
                            await websocket.send(json.dumps({"type": "collect_error", "message": "EEG设备未连接"}))
                            continue
                        # 创建保存目录
                        import datetime as _dt
                        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        self._collect_root = os.path.join(BASE_DIR, "data_self_browser", ts)
                        for lbl in range(4):
                            os.makedirs(os.path.join(self._collect_root, str(lbl + 1)), exist_ok=True)
                        self._collect_counter = {0: 0, 1: 0, 2: 0, 3: 0}
                        self._collect_active = True
                        log(f"[COLLECT] 数据采集已启动，保存到 {self._collect_root}")
                        await websocket.send(json.dumps({"type": "collect_started", "status": "ready"}))
                        continue

                    if msg_type == "collect_step":
                        if not getattr(self, '_collect_active', False):
                            await websocket.send(json.dumps({"type": "collect_error", "message": "采集未启动"}))
                            continue
                        expected_dir = data.get("direction")
                        if expected_dir not in ["up", "down", "left", "right"]:
                            await websocket.send(json.dumps({"type": "collect_error", "message": "无效方向"}))
                            continue
                        label = {"up": 0, "down": 1, "left": 2, "right": 3}[expected_dir]

                        # ---- PsychoPy 刺激流程 ----
                        if self.stim_clients:
                            await self._broadcast_stim({"type": "stim_target", "direction": expected_dir})
                            await self._broadcast_stim({"type": "stim_phase", "phase": "index"})
                            await asyncio.sleep(1.0)
                            await self._broadcast_stim({"type": "stim_phase", "phase": "stimulus", "direction": expected_dir})

                        # 记录起点，轮询等待—Trigger可能延迟到达
                        start = self.acq.get_sample_count()
                        import time as _time
                        deadline = _time.time() + 5.0
                        full = None
                        trigger_ch = None
                        onset = 0
                        while _time.time() < deadline:
                            await asyncio.sleep(0.05)
                            end = self.acq.get_sample_count()
                            if end - start < 200:  # 数据还不够，继续等
                                continue
                            with self.acq._lock_full:
                                full = np.array(self.acq.eeg_buffer_full[start:end], dtype=np.float64).T
                            trigger_ch = full[-1, :]
                            for i in range(len(trigger_ch) - 1):
                                if trigger_ch[i] < 0.5 and trigger_ch[i+1] >= 0.5:
                                    onset = i + 1
                                    break
                            if onset > 0 and onset + 500 <= full.shape[1]:
                                break  # 找到Trigger且数据够
                        if onset == 0 or onset + 500 > full.shape[1]:
                            log(f"[COLLECT] 数据长度不足: onset={onset} total={full.shape[1] if full is not None else 'N/A'}")
                            await websocket.send(json.dumps({"type": "collect_error", "message": "数据长度不足"}))
                            continue
                        uniq = np.unique(trigger_ch[:100])
                        log(f"[COLLECT] Trigger通道: 唯一值={uniq.tolist()}, onset={onset}, 范围=[{trigger_ch.min():.1f},{trigger_ch.max():.1f}]")
                        # 提取目标通道+Trigger (对齐离线实验: 14EEG + Trigger)
                        ch_idx = self.acq.channel_indices + [64]
                        raw = full[ch_idx, onset:onset + 500]  # (15, 500)
                        idx = self._collect_counter[label]
                        fname = os.path.join(self._collect_root, str(label + 1), f"browser_trial_{idx:04d}.npy")
                        np.save(fname, raw)      # (15,500): 14EEG + Trigger
                        self._collect_counter[label] += 1
                        log(f"[COLLECT] 已保存: {expected_dir} #{idx} → {fname}  shape={raw.shape}")
                        await websocket.send(json.dumps({"type": "collect_done", "direction": expected_dir, "index": idx}))
                        continue

                    if msg_type == "stop_collect":
                        log("[HANDLER] 收到 stop_collect")
                        self._collect_active = False
                        total = sum(self._collect_counter.values())
                        log(f"[COLLECT] 采集结束，共 {total} 试次: {dict(self._collect_counter)}")
                        await self._broadcast_stim({"type": "stim_stop"})
                        await self._stop_all()
                        await websocket.send(json.dumps({"type": "collect_stopped", "total": total}))
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

                        next_fn, skip_stale = _make_next_acq_sample(self.acq)
                        self._skip_stale = skip_stale
                        self.engine.data_source_callback = next_fn
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
        for client in list(self.clients):
            try:
                await client.send(msg)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def _broadcast_stim(self, data):
        """向 PsychoPy 刺激窗口广播消息。"""
        if not self.stim_clients:
            return
        msg = json.dumps(data)
        stale = []
        for client in list(self.stim_clients):
            try:
                await client.send(msg)
            except Exception:
                stale.append(client)
        for client in stale:
            self.stim_clients.discard(client)

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