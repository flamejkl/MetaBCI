# main.py
print("MAIN FILE =", __file__, flush=True)
import sys
import signal
import threading
from websocket_server import get_websocket_server

_stop_event = threading.Event()

def signal_handler(sig, frame):
    global _stop_event
    print("\n收到退出信号，正在退出...")
    _stop_event.set()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    ws = get_websocket_server()
    ws.start()
    print("WebSocket 服务器已启动，等待前端连接...")
    try:
        _stop_event.wait()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ws.stop()
        print("程序已退出")

if __name__ == "__main__":
    main()