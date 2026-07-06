# main.py
print("MAIN FILE =", __file__, flush=True)
import sys
# 关键：将 metabci 的父目录加入搜索路径
sys.path.insert(0, r"D:\pyproject\MetaBCI")

import signal
import time
from websocket_server import get_websocket_server

running = True

def signal_handler(sig, frame):
    global running
    print("\n收到退出信号，正在退出...")
    running = False

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    ws = get_websocket_server()
    ws.start()
    print("WebSocket 服务器已启动，等待前端连接...")
    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ws.stop()
        print("程序已退出")

if __name__ == "__main__":
    main()