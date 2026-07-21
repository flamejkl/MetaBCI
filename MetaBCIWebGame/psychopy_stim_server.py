# psychopy_stim_server.py
# -*- coding: utf-8 -*-
"""
PsychoPy SSVEP 刺激窗口 — 替代浏览器 Canvas 渲染。
通过 WebSocket 接收指令，提供与离线实验一致的刺激质量。
"""
import sys, os, json, time, threading, asyncio
import numpy as np
import websockets
import serial
import serial.tools.list_ports

# PsychoPy imports
from psychopy import visual, core, monitors, event

# ---- 配置 ----
STIM_FREQS = [8.25, 11.0, 13.75, 16.5]
PHASES = [0.0, 0.5, 1.0, 1.5]           # up, down, left, right (π 为单位)
DIRS = ['up', 'down', 'left', 'right']
FPS = 165.0                              # 显示器刷新率
BLOCK_RATIO = 0.12                       # 块视觉宽度 = 12% 屏宽 (stim_length)
POS_RATIO = 0.15                          # 块定位宽度 = 15% 屏宽 (block_ratio)
GAP_RATIO = 0.10                          # 块间额外间隙 = 10% 屏宽
WS_URL = "ws://localhost:8765"

# ---- 全局状态 ----
stim_flashing = False
stim_start_time = 0.0
current_target = None
collect_phase = None                    # 'preview'|'index'|'rest'|'stimulus'
cue_deadline = 0.0                      # 提示高亮截至时间（秒）
trigger_ser: serial.Serial = None       # TriggerBox 串口句柄
DIR_TO_TRIGGER = {'up': 1, 'down': 2, 'left': 3, 'right': 4}


def detect_triggerbox():
    """自动扫描 COM 口检测 TriggerBox，返回端口名或 None。"""
    for p in serial.tools.list_ports.comports():
        try:
            ser = serial.Serial(port=p.device, baudrate=115200,
                                bytesize=8, stopbits=1, parity='N', timeout=0.5)
            ser.write(bytes([0x01, 0x04, 0x00, 0x00]))
            time.sleep(0.1)  # 等足够长时间让TriggerBox响应
            resp = ser.read(50)
            ser.close()
            if resp and b'TriggerBox.Titing' in resp:
                return p.device
        except Exception:
            continue
    return None


def init_trigger():
    """初始化 TriggerBox 硬件打标。"""
    global trigger_ser
    port = detect_triggerbox()
    if port:
        trigger_ser = serial.Serial(port=port, baudrate=115200,
                                     bytesize=8, stopbits=1, parity='N', timeout=0.1)
        print(f'[Trigger] TriggerBox 已连接: {port}')
        return True
    print('[Trigger] 未检测到 TriggerBox，使用软件标记')
    return False


def send_trigger(code):
    """向 TriggerBox 发送触发编码 (1-5)。与离线实验命令格式一致。"""
    global trigger_ser
    if trigger_ser is None:
        print(f"[Trigger] 未连接，无法发送 code={code}")
        return
    try:
        cmd = bytes([0x01, 0xE1, 0x01, 0x00, code])
        trigger_ser.write(cmd)
        print(f"[Trigger] 已发送 code={code}")
    except Exception as e:
        print(f"[Trigger] 发送失败: {e}")


def create_window():
    """创建底部刺激窗口 — 置顶在浏览器下方。"""
    from psychopy import monitors as mon
    WINDOW_HEIGHT_RATIO = 0.25
    m = mon.Monitor('stimMonitor', width=53, distance=60, verbose=False)
    m.setSizePix([1920, 1080])
    scr_width, scr_height = m.getSizePix()
    win_height = int(scr_height * WINDOW_HEIGHT_RATIO)
    win = visual.Window(
        monitor=m,
        size=[scr_width, win_height],
        pos=[0, scr_height - win_height],
        color=(-1, -1, -1), colorSpace='rgb',
        fullscr=False,
        screen=0,
        units='pix',
        winType='pyglet',
        allowGUI=False,
        waitBlanking=True,
    )
    try:
        import ctypes
        hwnd = getattr(win.winHandle, '_hwnd', getattr(win.winHandle, 'handle', None))
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, scr_height-win_height, scr_width, win_height, 0x0001)
    except Exception:
        pass
    return win


def create_blocks(win):
    """创建 4 个闪烁刺激块，布局与 run_ssvep_experiment.py 一致。"""
    sw, sh = win.size
    bw = int(sw * BLOCK_RATIO)                       # 视觉块 = 12%屏宽 = 230px
    pos_w = int(sw * POS_RATIO)                       # 定位块 = 15%屏宽 = 288px
    gap = int(sw * GAP_RATIO)                         # 间隙 = 10%屏宽 = 192px
    total_w = 4 * pos_w + 3 * gap                     # = 1152+576 = 1728px
    start_x = -total_w / 2 + pos_w / 2                # 匹配实验 create_stimulus_positions
    step = pos_w + gap                                 # 中心距 = 288+192 = 480px

    blocks = []
    for i, (freq, phase, label) in enumerate(zip(STIM_FREQS, PHASES, DIRS)):
        x = start_x + i * step
        y = 0  # 垂直居中
        rect = visual.Rect(
            win, width=bw, height=bw,
            fillColor=[1, 1, 1], fillColorSpace='rgb',
            lineColor=None,
            pos=[x, y],
            autoLog=False,
        )
        label_text = visual.TextStim(
            win, text={'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[label],
            font='Arial', bold=True,
            color=[-1, -1, -1], colorSpace='rgb',
            height=bw * 0.35,
            pos=[x, y],
            autoLog=False,
        )
        blocks.append({
            'rect': rect, 'label': label_text,
            'freq': freq, 'phase': phase * np.pi,
            'dir': label, 'x': x, 'y': y, 'size': bw,
        })
    return blocks


def draw_preview(win, blocks):
    """预览阶段：所有块白色常亮。"""
    for b in blocks:
        b['rect'].fillColor = (1, 1, 1)
        b['rect'].draw()
        b['label'].color = (-1, -1, -1)
        b['label'].draw()


def draw_index(win, blocks, target_dir):
    """提示阶段：目标块亮白，其余暗灰。"""
    for b in blocks:
        is_target = (b['dir'] == target_dir)
        c = 1.0 if is_target else 0.25
        b['rect'].fillColor = (c, c, c)
        b['rect'].draw()
        b['label'].color = (-1, -1, -1) if is_target else (1, 1, 1)
        b['label'].draw()
    # 红色三角箭头 — 在目标块邻接间隙中，指向目标块
    b_up = next(b for b in blocks if b['dir'] == 'up')
    b_down = next(b for b in blocks if b['dir'] == 'down')
    b_left = next(b for b in blocks if b['dir'] == 'left')
    b_right = next(b for b in blocks if b['dir'] == 'right')
    tri_size = min(b_up['size'], b_up['size']) * 0.8
    tri_y = b_up['y']
    # 左间隙(↑↓之间)、右间隙(←→之间)，箭头指向目标
    left_gap_x = (b_up['x'] + b_up['size']/2 + b_down['x'] - b_down['size']/2) / 2
    right_gap_x = (b_left['x'] + b_left['size']/2 + b_right['x'] - b_right['size']/2) / 2
    gap_map = {
        'up':    (left_gap_x,  90.0),   # 左间隙，箭头指左(↑)
        'down':  (left_gap_x,  270.0),  # 左间隙，箭头指右(↓)
        'left':  (right_gap_x, 90.0),   # 右间隙，箭头指左(←)
        'right': (right_gap_x, 270.0),  # 右间隙，箭头指右(→)
    }
    tri_x, ori = gap_map[target_dir]
    tri = visual.TextStim(
        win, text='⯆', font='Arial', bold=True,
        color=(1.0, -1.0, -1.0), colorSpace='rgb',
        height=tri_size,
        pos=[tri_x, tri_y],
        ori=ori,
        autoLog=False,
    )
    tri.draw()


def draw_rest(win, blocks):
    """休息阶段：十字准星。"""
    cross_h = visual.ShapeStim(win, vertices=[(-20, 0), (20, 0)],
                               lineWidth=3, lineColor='white', closeShape=False, autoLog=False)
    cross_v = visual.ShapeStim(win, vertices=[(0, 20), (0, -20)],
                               lineWidth=3, lineColor='white', closeShape=False, autoLog=False)
    cross_h.draw()
    cross_v.draw()


def draw_flash(win, blocks, elapsed):
    """闪烁阶段：正弦波调制（与训练实验完全一致）。"""
    for b in blocks:
        val = 0.7 + 0.3 * np.sin(2 * np.pi * b['freq'] * elapsed + b['phase'])  # 0.4~1.0
        b['rect'].fillColor = (val, val, val)
        b['rect'].draw()
        b['label'].color = (-1, -1, -1) if val > 0.5 else (1, 1, 1)
        b['label'].draw()


# ---- 主渲染循环 ----
def render_loop(win, blocks):
    """主循环：根据全局状态渲染刺激块。"""
    global stim_flashing, stim_start_time, current_target, collect_phase, cue_deadline
    clock = core.Clock()
    trial_start = 0.0

    while True:
        t = time.perf_counter()

        # stim_target 短暂高亮（eval快速模式）：自动过期回到闪烁
        if collect_phase is None and cue_deadline > 0 and t > cue_deadline:
            cue_deadline = 0.0

        if collect_phase == 'preview':
            draw_preview(win, blocks)
        elif collect_phase == 'index' and current_target:
            draw_index(win, blocks, current_target)
        elif collect_phase == 'rest':
            draw_rest(win, blocks)
        elif collect_phase == 'stimulus':
            elapsed = t - stim_start_time
            draw_flash(win, blocks, elapsed)
        elif cue_deadline > 0 and current_target:
            draw_index(win, blocks, current_target)
        elif stim_flashing:
            elapsed = t - stim_start_time
            draw_flash(win, blocks, elapsed)
        else:
            for b in blocks:
                b['rect'].fillColor = (0.5, 0.5, 0.5)
                b['rect'].draw()
                b['label'].color = (-1, -1, -1)
                b['label'].draw()

        win.flip()

        # 按键退出
        keys = event.getKeys(['escape', 'q'])
        if keys:
            break

    win.close()
    core.quit()


# ---- WebSocket 客户端 ----
async def ws_client():
    """连接后端 WebSocket，接收指令。"""
    global stim_flashing, stim_start_time, current_target, collect_phase, cue_deadline

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("[PsychoPy] 已连接 WebSocket")
                await ws.send(json.dumps({"type": "stim_register"}))
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        t = data.get("type")

                        if t == "stim_start":
                            stim_flashing = True
                            stim_start_time = time.perf_counter()
                            # 发送开始触发 (编码1=通用开始)
                            send_trigger(1)
                            print("[PsychoPy] 刺激开始")

                        elif t == "stim_stop":
                            stim_flashing = False
                            collect_phase = None
                            current_target = None
                            # 发送结束触发 (编码5=试次结束)
                            send_trigger(5)
                            print("[PsychoPy] 刺激停止")

                        elif t == "stim_phase":
                            phase = data.get("phase")
                            collect_phase = phase
                            if phase == 'stimulus':
                                stim_flashing = True
                                stim_start_time = time.perf_counter()
                                # 方向触发: 从后端传来的 direction 字段
                                direction = data.get("direction", "")
                                code = DIR_TO_TRIGGER.get(direction, 1)
                                send_trigger(code)
                            else:
                                stim_flashing = False
                                if phase == 'rest':
                                    send_trigger(5)  # 试次结束
                            print(f"[PsychoPy] 阶段: {phase}")

                        elif t == "stim_target":
                            current_target = data.get("direction")
                            # 短暂高亮 0.8s 提示用户注视目标方向
                            cue_deadline = time.perf_counter() + 0.8
                            # 发送方向触发
                            code = DIR_TO_TRIGGER.get(current_target, 1)
                            send_trigger(code)
                            print(f"[PsychoPy] 目标: {current_target} (cue 0.8s, trigger={code})")

                    except Exception as e:
                        print(f"[PsychoPy] 消息处理错误: {e}")
        except Exception as e:
            print(f"[PsychoPy] WebSocket 连接断开: {e}, 2秒后重连...")
            await asyncio.sleep(2)


def main():
    print("=" * 50)
    print("PsychoPy SSVEP 刺激窗口")
    print(f"频率: {STIM_FREQS} Hz")
    print(f"刷新率: {FPS} Hz")
    print(f"WebSocket: {WS_URL}")
    print("=" * 50)

    win = create_window()
    blocks = create_blocks(win)
    print(f"窗口: {win.size[0]}×{win.size[1]} px, 块: {blocks[0]['size']}px")

    # 初始化 TriggerBox 硬件打标
    init_trigger()

    # WebSocket 在后台线程运行
    loop = asyncio.new_event_loop()

    def run_ws():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_client())

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # 主线程运行 PsychoPy 渲染循环
    render_loop(win, blocks)


if __name__ == "__main__":
    main()
