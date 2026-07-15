# -*- coding: utf-8 -*-
# License: MIT License
"""
Online SSVEP streaming engine for continuous brain-control.

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
    await engine.start()
"""
import asyncio
import time
import numpy as np
from typing import Optional, Callable, Awaitable, Tuple


class ContinuousStreamingEngine:
    """Async streaming engine for continuous EEG decoding.

    Parameters
    ----------
    model : object
        A trained classifier with a ``transform`` method.
    decoder : object
        A decoder with ``feed(sample)``, ``reset()``, and optionally ``slide()``.
    occipital_indices : list of int
        Indices of occipital channels in the incoming 14-ch data.
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
        self._stop_event = asyncio.Event()

        # Data source callback → (ndarray | None, bool, dict)
        self.data_source_callback: Optional[
            Callable[[], Tuple[Optional[np.ndarray], bool, dict]]
        ] = None
        self.emit_callback: Optional[Callable[[dict], Awaitable[None]]] = None

        self.context = {"expected_dir": None, "msg_type": None, "trial_started": False}
        self.current_extra = {}
        self.frame_count = 0
        self.decision_count = 0
        self.last_decision_time = 0.0

    # ------------------------------------------------------------------
    async def _continuous_loop(self):
        self._running = True
        self._stop_event.clear()

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
                        for sample in data_chunk:
                            if sample.shape[0] == 14:
                                sample = sample[self.occipital_indices]
                            decision, conf, cur_t = self.decoder.feed(sample)
                            if decision is not None and self.state != self.State.IDLE:
                                await self._on_decision(decision, conf, cur_t)

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0, 0.004 - elapsed))

    async def _on_decision(self, decision, conf, current_time):
        self.decision_count += 1
        self.last_decision_time = current_time

        if self.state == self.State.REALTIME:
            command = ["up", "down", "left", "right"][decision]
            if self.emit_callback:
                await self.emit_callback({
                    "type": "realtime_command",
                    "command": command,
                    "confidence": conf,
                    "all_confidences": [0.0] * 4,
                })
        elif self.state in (self.State.DEMO, self.State.EVAL):
            expected = self.context.get("expected_dir")
            msg_type = self.context.get("msg_type", "eval_result")
            if expected is not None:
                decoded_dir = ["up", "down", "left", "right"][decision]
                match = (decoded_dir == expected)
                msg = {
                    "type": msg_type,
                    "decoded": decoded_dir,
                    "expected": expected,
                    "match": match,
                    "timeout": False,
                    "confidence": conf,
                    "all_confidences": [0.0] * 4,
                }
                if 'filename' in self.current_extra:
                    msg['filename'] = self.current_extra['filename']
                if self.emit_callback:
                    await self.emit_callback(msg)
                self.context["expected_dir"] = None
                self.context["msg_type"] = None

        if self.state != self.State.REALTIME:
            self.decoder.reset()
        else:
            if hasattr(self.decoder, 'slide'):
                self.decoder.slide()

    # ------------------------------------------------------------------
    async def start(self):
        if self._running:
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

    def set_mode(self, state: str, expected_dir=None, msg_type=None):
        self.state = state
        self.context["expected_dir"] = expected_dir
        self.context["msg_type"] = msg_type
