# main.py
print("MAIN FILE =", __file__, flush=True)
import sys, os, signal, threading
from websocket_server import get_websocket_server

_stop_event = threading.Event()

def signal_handler(sig, frame):
    global _stop_event
    print("\n收到退出信号，正在退出...")
    _stop_event.set()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 静态文件已移入 MetaBCI 框架: metabci/webgame/static/
    from metabci.webgame import STATIC_DIR as FRONTEND_DIR
    print(f"前端目录: {FRONTEND_DIR}")

    ws = get_websocket_server()
    ws.start()
    print("WebSocket 服务器已启动，等待前端连接...")
    print(f"浏览器打开: http://localhost:8765/static/index.html  (需另启 HTTP 服务)")
    print(f"  cd {FRONTEND_DIR} && python -m http.server 8000")
    try:
        _stop_event.wait()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ws.stop()
        print("程序已退出")

if __name__ == "__main__":
    main()