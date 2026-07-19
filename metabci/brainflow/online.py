# -*- coding: utf-8 -*-
# License: MIT License
"""
Online streaming engine for continuous brain-control decoding.

Provides ContinuousStreamingEngine — an asyncio-based engine that feeds
real-time EEG samples through a decoder (e.g. GrowingWindowDecoder) and
emits decisions via a user-supplied async callback.  Supports DEMO (offline
playback), EVAL (online evaluation with labels), and REALTIME (continuous
brain-control) modes.

Reference
---------
.. code-block:: python

    from metabci.brainflow.online import ContinuousStreamingEngine

    engine = ContinuousStreamingEngine(model, decoder)
    engine.emit_callback = my_async_send
    engine.data_source_callback = my_data_source
    await engine.start()
"""
import asyncio
import concurrent.futures
import queue
import threading
import time
import numpy as np
from typing import Optional, Callable, Awaitable, Tuple


class ContinuousStreamingEngine:
    """Async streaming engine with threaded decoder execution.

    Decoder inference runs in a daemon thread so the asyncio event loop
    stays responsive for WebSocket I/O and stimulus timing.

    Parameters
    ----------
    model : object (unused directly; decoder handles inference).
    decoder : object
        A decoder with ``feed(sample) → (decision, conf, t)``, ``reset()``,
        and optionally ``slide()``.
    occipital_indices : list of int
        Indices of occipital channels in 14-ch data.
    sample_rate : int
        EEG sampling rate in Hz.
    """

    class State:
        IDLE = "IDLE"
        DEMO = "DEMO"
        EVAL = "EVAL"
        REALTIME = "REALTIME"

    def __init__(self, model, decoder, occipital_indices=None, sample_rate=250):
        if occipital_indices is None:
            occipital_indices = [2, 3, 4, 5, 6, 7, 8, 9]
        self.model = model
        self.decoder = decoder
        self.occipital_indices = occipital_indices
        self.sample_rate = sample_rate

        self.state = self.State.IDLE
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        self.data_source_callback: Optional[
            Callable[[], Tuple[Optional[np.ndarray], bool, dict]]
        ] = None
        self.emit_callback: Optional[Callable[[dict], Awaitable[None]]] = None

        self.context = {"expected_dir": None, "msg_type": None}
        self.current_extra: dict = {}
        self.frame_count = 0
        self.decision_count = 0
        self.last_decision_time = 0.0

        # Threading: decoder runs in daemon thread, results posted to queue
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_future: Optional[asyncio.Future] = None

    # ------------------------------------------------------------------ #
    #  Main loop (asyncio event loop — non-blocking I/O)
    # ------------------------------------------------------------------ #
    async def _continuous_loop(self):
        import traceback
        self._running = True
        while self._running:
            loop_start = time.perf_counter()
            try:
                if self.data_source_callback and self._pending_future is None:
                    result = self.data_source_callback()
                    if result is not None:
                        data_chunk, is_new_trial, extra_info = result
                        if data_chunk is not None:
                            if is_new_trial and self.state in (self.State.DEMO, self.State.EVAL):
                                self.decoder.reset()
                                self.current_extra = extra_info or {}
                            # Offload blocking decode to thread, keep event loop free
                            if self.state != self.State.IDLE:
                                loop = asyncio.get_running_loop()
                                self._pending_future = loop.run_in_executor(
                                    self._executor,
                                    self._decode_trial,
                                    data_chunk,
                                )

                # Check if threaded decode completed
                if self._pending_future is not None and self._pending_future.done():
                    try:
                        decision, conf, cur_t = self._pending_future.result()
                    except Exception as e:
                        traceback.print_exc()
                        decision, conf, cur_t = None, 0.0, 0.0
                    self._pending_future = None
                    if decision is not None:
                        await self._on_decision(decision, conf, cur_t)
            except Exception:
                traceback.print_exc()

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0, 0.004 - elapsed))

    # ------------------------------------------------------------------ #
    #  Blocking decode (runs in thread pool)
    # ------------------------------------------------------------------ #
    def _decode_trial(self, data_chunk):
        """Feed a full trial through the decoder.  Called from a worker thread."""
        import traceback
        decision = None
        conf = 0.0
        cur_t = 0.0
        try:
            for i, sample in enumerate(data_chunk):
                if sample.shape[0] == 14:
                    sample = sample[self.occipital_indices]
                d, c, t = self.decoder.feed(sample)
                if d is not None:
                    decision, conf, cur_t = d, c, t
                    break
        except Exception:
            # 解码器异常不应导致整个引擎崩溃
            traceback.print_exc()
        return decision, conf, cur_t

    # ------------------------------------------------------------------ #
    #  Decision handling
    # ------------------------------------------------------------------ #
    async def _on_decision(self, decision, conf, current_time):
        self.decision_count += 1
        self.last_decision_time = current_time

        if self.state == self.State.REALTIME:
            command = ["up", "down", "left", "right"][decision]
            all_conf = self._build_confidences(decision, conf)
            if self.emit_callback:
                await self.emit_callback({
                    "type": "realtime_command",
                    "command": command,
                    "confidence": conf,
                    "all_confidences": all_conf,
                })

        elif self.state in (self.State.DEMO, self.State.EVAL):
            expected = self.context.get("expected_dir")
            msg_type = self.context.get("msg_type", "eval_result")
            if expected is not None:
                decoded_dir = ["up", "down", "left", "right"][decision]
                match = (decoded_dir == expected)
                all_conf = self._build_confidences(decision, conf)
                msg = {
                    "type": msg_type,
                    "decoded": decoded_dir,
                    "expected": expected,
                    "match": match,
                    "timeout": False,
                    "confidence": conf,
                    "all_confidences": all_conf,
                    "decision_time": current_time,
                }
                if 'filename' in self.current_extra:
                    msg['filename'] = self.current_extra['filename']
                if self.emit_callback:
                    await self.emit_callback(msg)
                self.context["expected_dir"] = None
                self.context["msg_type"] = None

        # Reset or slide decoder after each decision
        if self.state != self.State.REALTIME:
            self.decoder.reset()
        elif hasattr(self.decoder, 'slide'):
            self.decoder.slide()

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self):
        if self._running:
            return
        # 重建 executor，防止复用已 shutdown 的线程池
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._loop_task = asyncio.create_task(self._continuous_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        self.state = self.State.IDLE
        self.context = {"expected_dir": None, "msg_type": None}
        self._pending_future = None
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self.decoder.reset()
        self._executor.shutdown(wait=False)

    def set_mode(self, state: str, expected_dir=None, msg_type=None):
        self.state = state
        self.context["expected_dir"] = expected_dir
        self.context["msg_type"] = msg_type

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _build_confidences(self, decision, conf):
        """Return softmax-normalised 4-class confidence list."""
        scores = getattr(self.decoder, '_last_scores', None)
        if scores is not None:
            s = np.array(scores, dtype=np.float64)
            s -= s.max()
            exp_s = np.exp(s)
            return (exp_s / exp_s.sum()).tolist()

        # Fallback: distribute remaining mass evenly
        result = [0.0] * 4
        result[decision] = conf
        rem = max(0.0, (1.0 - conf) / 3.0)
        for i in range(4):
            if i != decision:
                result[i] = rem
        return result
