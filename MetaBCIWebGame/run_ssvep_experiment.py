# run_ssvep_experiment.py
import os
import time
import numpy as np
import scipy.io as sio
from datetime import datetime
from psychopy import visual, monitors, core, event
from metabci.brainstim.paradigm import SSVEP
from data_acquisition import DataAcquisition
import serial
import serial.tools.list_ports

# ==================== 实验参数 ====================
STIM_FREQS = [8.25, 11.0, 13.75, 16.5]   # 与165Hz完美匹配
STIM_DURATION = 2.0                            # 闪烁时长（秒）
PREVIEW_DURATION = 1                           # 预览时长（秒，常亮不闪烁）
REST_DURATION = 1.0                            # 休息时长（秒）
N_REPEATS = 40                                 # 每个方向重复次数
MODE = 'real'                                  # 'simulate' 或 'real'
SERIAL_PORT = 'COM3'                           # 备用，用于自动检测
SAVE_ROOT = r"D:\pyproject\MetaBCI\MetaBCIWebGame\data"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ==================== TriggerBox 自动检测 ====================
def detect_triggerbox():
    """
    自动扫描所有可用的 COM 口，尝试发送 DCP 命令 01 04 00 00，
    若收到包含 'TriggerBox.Titing' 的回复，则返回该端口号，否则返回 None。
    """
    import time
    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        try:
            ser = serial.Serial(
                port=p.device,
                baudrate=115200,
                bytesize=8,
                stopbits=1,
                parity='N',
                timeout=0.5
            )
            ser.reset_input_buffer()
            cmd = bytes([0x01, 0x04, 0x00, 0x00])
            ser.write(cmd)
            time.sleep(0.05)
            response = ser.read(ser.in_waiting or 30)
            ser.close()
            if response:
                try:
                    resp_str = response.decode('ascii', errors='ignore')
                    if 'TriggerBox.Titing' in resp_str:
                        print(f"[TriggerBox] 检测到有效设备，端口 {p.device}")
                        return p.device
                except:
                    pass
                hex_resp = response.hex().upper()
                if '54726967676572426F782E546974696E67' in hex_resp:
                    print(f"[TriggerBox] 检测到有效设备（hex匹配），端口 {p.device}")
                    return p.device
        except Exception:
            continue
    print("[TriggerBox] 未检测到任何有效 TriggerBox，将使用软件标记")
    return None

# ==================== 辅助函数 ====================
def create_stimulus_positions(win, block_ratio=0.15, gap_ratio=0.1):
    """返回方块的中心坐标数组"""
    screen_width, screen_height = win.size
    block_width = int(screen_width * block_ratio)
    block_height = block_width
    total_width = 4 * block_width + 3 * gap_ratio * screen_width
    start_x = -total_width / 2 + block_width / 2
    positions = []
    for i in range(4):
        x = start_x + i * (block_width + gap_ratio * screen_width)
        y = 0
        positions.append((x, y))
    return np.array(positions)

def show_preview(win, positions, block_size, duration, index_stimulus=None):
    rects = []
    for pos in positions:
        rect = visual.Rect(win, width=block_size, height=block_size,
                           fillColor=[1,1,1], lineColor=None, pos=pos)
        rects.append(rect)
    start = time.time()
    while time.time() - start < duration:
        for rect in rects:
            rect.draw()
        if index_stimulus is not None:
            index_stimulus.draw()
        win.flip()
        if event.getKeys(['escape','q','space']):
            return False
    return True

def show_rest(win, duration):
    hline = visual.ShapeStim(win, vertices=[(-15, 0), (15, 0)], lineWidth=3, lineColor='white', closeShape=False)
    vline = visual.ShapeStim(win, vertices=[(0, 15), (0, -15)], lineWidth=3, lineColor='white', closeShape=False)
    start = time.time()
    while time.time() - start < duration:
        hline.draw()
        vline.draw()
        win.flip()
        if event.getKeys(['escape','q','space']):
            return False
    return True

def wait_for_space(win):
    msg = visual.TextStim(win, text="按空格键开始实验...", color='white', height=40, units='pix')
    while True:
        msg.draw()
        win.flip()
        keys = event.getKeys(['space', 'escape', 'q'])
        if 'space' in keys:
            return True
        if 'escape' in keys or 'q' in keys:
            return False

# ==================== 主实验流程 ====================
def run_experiment():
    print("="*60)
    print("SSVEP 四方向采集实验（预览 + 闪烁）")
    print(f"配置模式: {MODE}, 每方向重复: {N_REPEATS}, 总试次: {4*N_REPEATS}")
    print(f"保存路径: {SAVE_ROOT}")
    print("="*60)

    # ---- 1. 检测 TriggerBox ----
    trigger_port = detect_triggerbox() if MODE == 'real' else None
    use_hardware_trigger = (trigger_port is not None)

    # ---- 2. 初始化 EEG 采集 ----
    run_mode = MODE
    acq = None
    sim_full_data = [] if run_mode == 'simulate' else None
    sim_events = [] if run_mode == 'simulate' else None

    if run_mode == 'real':
        print("\n正在尝试连接真实 EEG 设备...")
        try:
            acq = DataAcquisition(
                mode='real',
                neuracle_ip='127.0.0.1',
                neuracle_port=8712,
                srate=250,
                num_chans=14
            )
            if acq.connect():
                acq.start_acquisition()
                acq.reset_buffer()
                print("✅ Neuracle EEG 设备连接成功，开始采集（250 Hz）")
                eeg_ok = True
            else:
                print("❌ Neuracle EEG 设备连接失败")
                eeg_ok = False
        except Exception as e:
            print(f"❌ Neuracle EEG 设备连接异常: {e}")
            eeg_ok = False

        if not eeg_ok:
            print("\⚠️ 真实 EEG 设备未连接，数据采集将无法进行。")
            choice = input("是否继续以模拟模式运行？(y/n): ").strip().lower()
            if choice == 'y':
                print("已切换到模拟模式。")
                run_mode = 'simulate'
                sim_full_data = []
                sim_events = []
                if acq:
                    acq.stop_acquisition()
                    acq = None
            else:
                print("用户选择退出。")
                return
        else:
            print("✅ EEG 就绪。")
            if use_hardware_trigger:
                print(f"✅ 使用硬件 TriggerBox（端口 {trigger_port}）进行触发标记。")
                global trigger_ser
                trigger_ser = serial.Serial(
                    port=trigger_port,
                    baudrate=115200,
                    bytesize=8,
                    stopbits=1,
                    parity='N',
                    timeout=0.1
                )
            else:
                print("ℹ️ 未检测到 TriggerBox，使用软件时间标记。")

    else:
        print("模拟模式：无硬件，将生成模拟数据。")
        sim_full_data = []
        sim_events = []

    if hasattr(acq, 'server') and acq.server is not None:
        all_ch_names = acq.server.channelNames
        print(f"所有通道名称（共{len(all_ch_names)}个）:")
        for i, name in enumerate(all_ch_names):
            print(f"{i}: {name}")

    # ---- 3. 创建窗口和刺激对象 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_ROOT, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    mon = monitors.Monitor('tempMonitor', width=53, distance=60, verbose=False)
    win = visual.Window(monitor=mon, color='black', colorSpace='rgb', fullscr=True,
                        units='pix', winType='pyglet')
    fps = 165.0
    print(f"刷新率: {fps:.2f} Hz")

    # 创建 SSVEP 对象
    stim_pos = create_stimulus_positions(win, 0.15, 0.1)
    ssvep = SSVEP(win=win)
    ssvep.config_pos(n_elements=4, rows=1, columns=4, stim_pos=stim_pos,
                     stim_length=int(win.size[0]*0.12), stim_width=int(win.size[0]*0.12))
    ssvep.config_index()
    ssvep.config_color(refresh_rate=fps,
                       stim_time=STIM_DURATION,
                       stimtype='sinusoid',
                       stim_color=[1,1,1],
                       freqs=STIM_FREQS,
                       phases=[0,0.5,1,1.5])
    flash_seq = ssvep.flash_stimuli

    # 试次序列
    trial_sequence = []
    for d in range(4):
        trial_sequence.extend([d] * N_REPEATS)
    np.random.shuffle(trial_sequence)

    # 等待开始
    if not wait_for_space(win):
        win.close()
        core.quit()
        return

    # ---- 4. 主循环 ----
    events_list = []                # 存储 (start_idx, end_idx, direction)
    block_size = int(win.size[0] * 0.12)
    trial_start_times = []
    interrupted = False

    # 辅助函数：发送硬件触发
    def send_trigger(value):
        if use_hardware_trigger and trigger_ser and trigger_ser.is_open:
            try:
                cmd = bytes([0x01, 0xE1, 0x01, 0x00, value])
                trigger_ser.write(cmd)
            except Exception as e:
                print(f"硬件触发发送失败: {e}")

    for trial_idx, stim_idx in enumerate(trial_sequence):
        print(f"\n试次 {trial_idx+1}/{len(trial_sequence)}: 目标 {stim_idx}")

        # 预览阶段
        target_pos = stim_pos[stim_idx]
        ssvep.index_stimuli.setPos((target_pos[0], target_pos[1] + block_size / 2 + 60))
        if not show_preview(win, stim_pos, block_size, PREVIEW_DURATION, ssvep.index_stimuli):
            interrupted = True
            break

        # ---- 刺激阶段 ----
        if run_mode == 'real' and acq:
            if use_hardware_trigger:
                send_trigger(stim_idx + 1)   # 1~4
            time.sleep(0.002)  # 让采集线程更新缓冲区
            start_idx = acq.get_sample_count()
        else:
            start_idx = sum([d.shape[1] for d in sim_full_data]) if sim_full_data else 0

        trial_start_times.append(time.time())

        # 播放闪烁序列
        for frame in flash_seq:
            frame.draw()
            win.flip()
            if event.getKeys(['escape','q','space']):
                interrupted = True
                break
        else:
            # 正常结束
            if run_mode == 'real' and acq:
                if use_hardware_trigger:
                    send_trigger(5)   # 结束标记
                end_idx = start_idx + int(STIM_DURATION * acq.srate) - 1
                events_list.append((start_idx, end_idx, stim_idx))
            else:
                srate = 250
                nchans = 14
                ns = int(STIM_DURATION * srate)
                eeg = np.random.randn(nchans, ns) * 1e-5
                trig = np.zeros((1, ns))
                trig[0,0] = stim_idx + 1
                trig[0,-1] = 5
                trial_data = np.vstack([eeg, trig])
                sim_full_data.append(trial_data)
                end_idx = start_idx + ns - 1
                sim_events.append((start_idx, end_idx, stim_idx))

            # 休息阶段
            if not show_rest(win, REST_DURATION):
                interrupted = True
                break
            continue
        if interrupted:
            break

    # ---- 5. 保存数据 ----
    if run_mode == 'real':
        if acq:
            print("正在提取 EEG 数据...")
            # 获取完整的65通道数据
            full_eeg = acq.get_all_data_with_trigger()
            if full_eeg.size == 0:
                print("警告：未采集到任何 EEG 数据，跳过保存。")
            else:
                # 获取通道名称
                all_ch_names = acq.server.channelNames if hasattr(acq, 'server') and acq.server else None
                if all_ch_names is None:
                    print("警告：无法获取通道名称，将使用默认索引。")
                    # 回退：保存全部65个通道（但会占用内存）
                    indices_to_save = list(range(full_eeg.shape[0]))
                    ch_names_to_save = [f"CH{i}" for i in range(full_eeg.shape[0])]
                else:
                    # 目标通道索引（14个EEG）
                    target_indices = acq.channel_indices
                    # Trigger 通道索引（如果存在）
                    trigger_idx = all_ch_names.index('Trigger') if 'Trigger' in all_ch_names else None
                    indices_to_save = target_indices.copy()
                    if trigger_idx is not None:
                        indices_to_save.append(trigger_idx)
                    # 对应的通道名称
                    ch_names_to_save = [all_ch_names[i] for i in indices_to_save]
                    print(f"保存通道索引: {indices_to_save}")
                    print(f"保存通道名称: {ch_names_to_save}")
                # 提取需要的通道
                eeg_to_save = full_eeg[indices_to_save, :]
                save_dict = {
                    'eeg_data': eeg_to_save,
                    'events': events_list,
                    'sample_rate': 250,
                    'channels': ch_names_to_save,
                    'trigger_mode': 'hardware' if use_hardware_trigger else 'software'
                }
                save_path = os.path.join(save_dir, "full_experiment.mat")
                sio.savemat(save_path, save_dict)
                print(f"完整数据已保存至 {save_path}")
                print(f"总样本数: {eeg_to_save.shape[1]}, 记录试次数: {len(events_list)}")
                print(f"保存通道数: {eeg_to_save.shape[0]}")
            acq.stop_acquisition()
        else:
            print("采集对象未初始化，无法保存数据")
    else:
        if sim_full_data:
            full_sim = np.concatenate(sim_full_data, axis=1)
            save_dict = {
                'eeg_data': full_sim[:-1, :],
                'trigger': full_sim[-1, :],
                'events': sim_events,
                'sample_rate': 250,
                'channels': ['Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO5', 'PO6', 'POz', 'P3', 'P4', 'P7', 'P8'],
                'trigger_mode': 'software'
            }
            save_path = os.path.join(save_dir, "full_simulated_experiment.mat")
            sio.savemat(save_path, save_dict)
            print(f"模拟数据已保存至 {save_path}")
        else:
            print("警告：没有生成任何模拟数据")

    # 关闭窗口和串口
    win.close()
    if use_hardware_trigger and trigger_ser and trigger_ser.is_open:
        trigger_ser.close()
    print("实验结束")
    return

if __name__ == "__main__":
    run_experiment()