# neuracle_trigger_enhanced.py
import serial
import serial.tools.list_ports
import time
from metabci.brainstim.utils import NeuraclePort

class EnhancedNeuraclePort(NeuraclePort):
    """
    继承自 NeuraclePort，增加设备验证、错误处理和响应读取功能。
    """
    def __init__(self, port_addr, baudrate=115200, auto_validate=True):
        """
        初始化串口，并可选自动验证设备。
        :param port_addr: 串口号，如 'COM3'
        :param baudrate: 波特率，默认115200
        :param auto_validate: 是否在初始化时自动验证设备
        """
        # 先不直接打开串口，先验证设备可用性
        self.port_addr = port_addr
        self.baudrate = baudrate
        self.port = None
        self._device_verified = False
        if auto_validate:
            self.validate_device()

    def _open_serial(self):
        """内部方法：打开串口（如果尚未打开）"""
        if self.port is None or not self.port.is_open:
            try:
                self.port = serial.Serial(port=self.port_addr, baudrate=self.baudrate, timeout=1)
                print(f"串口 {self.port_addr} 已打开")
            except Exception as e:
                raise RuntimeError(f"无法打开串口 {self.port_addr}: {e}")

    def check_online(self):
        """检查串口号是否在线（即是否存在该串口设备）"""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if p.device == self.port_addr:
                return True
        return False

    def validate_device(self):
        """
        验证设备是否可通信。
        尝试发送获取设备名称命令，并检查响应。
        """
        if not self.check_online():
            raise RuntimeError(f"串口 {self.port_addr} 不存在或未连接")
        self._open_serial()
        # 清空缓冲区
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        # 命令格式：deviceID=1, functionID=4 (DeviceNameGet), payload=0
        cmd = bytes([0x01, 0x04, 0x00, 0x00])
        self.port.write(cmd)
        # 等待并读取响应（4字节头 + 可变负载）
        try:
            header = self.port.read(4)
            if len(header) < 4:
                raise RuntimeError("设备无响应")
            device_id = header[0]
            function_id = header[1]
            payload_len = header[2] | (header[3] << 8)
            if device_id != 1 or function_id != 4:
                raise RuntimeError("设备响应格式错误")
            if payload_len > 0:
                resp = self.port.read(payload_len)
                device_name = resp.decode('utf-8', errors='ignore')
                print(f"设备名称: {device_name}")
            else:
                print("设备响应无负载")
            self._device_verified = True
        except Exception as e:
            raise RuntimeError(f"设备验证失败: {e}")

    def setData(self, label):
        """
        发送触发标记（重写父类方法，增加错误处理）
        """
        if not self._device_verified:
            # 如果未验证，尝试验证（或直接发送）
            try:
                self.validate_device()
            except Exception as e:
                print(f"警告：设备未验证，发送可能失败: {e}")
        try:
            super().setData(label)
        except Exception as e:
            print(f"发送标记 {label} 失败: {e}")
            # 可选：尝试重新打开串口
            self._reopen_serial()

    def _reopen_serial(self):
        """尝试重新打开串口"""
        if self.port:
            try:
                self.port.close()
            except:
                pass
            self.port = None
        self._open_serial()

    def closeSerial(self):
        """关闭串口"""
        if self.port and self.port.is_open:
            self.port.close()
            print("串口已关闭")

    def __del__(self):
        self.closeSerial()