from queue import Queue
import socket
import selectors
import threading

class MessageBroker:
    def __init__(self, host="localhost", port=9999):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.subscribers = {}
        self.message_queue = Queue()  # 消息队列
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(100)
        self.server.setblocking(False)
        self.sel.register(self.server, selectors.EVENT_READ, self.accept)

        # 启动异步消息处理线程
        threading.Thread(target=self.process_messages, daemon=True).start()

    def process_messages(self):
        """异步处理消息队列中的消息，并转发给订阅者"""
        while True:
            message = self.message_queue.get()
            if message is None:
                break  # 停止处理
            self.broadcast(message)

    def accept(self, server_sock):
        client_socket, addr = server_sock.accept()
        print(f"Connection from {addr}")
        client_socket.setblocking(False)
        self.sel.register(client_socket, selectors.EVENT_READ, self.handle_client)

    def handle_client(self, client_socket):
        try:
            data = client_socket.recv(1024).decode()
            if not data:
                self._close_connection(client_socket)
                return

            role, message = data.split("\n", 1)
            role = role.strip()

            if role == "PUBLISHER":
                print(f"Received message from publisher: {message}")
                # 将消息放入队列，异步处理
                self.message_queue.put(message)
            elif role == "SUBSCRIBER":
                with self.lock:
                    self.subscribers[client_socket] = Queue()
                print(f"新增订阅者，总订阅者数量: {len(self.subscribers)}")
            else:
                print("未知角色，关闭连接")
                self._close_connection(client_socket)
        except Exception as e:
            print(f"处理客户端时出错: {e}")
            self._close_connection(client_socket)

    def broadcast(self, message):
        """将消息发送给所有订阅者"""
        with self.lock:
            for subscriber, queue in self.subscribers.items():
                queue.put(message)  # 将消息放入每个订阅者的队列
                threading.Thread(target=self.send_message, args=(subscriber,), daemon=True).start()

    def send_message(self, subscriber):
        """异步发送消息"""
        try:
            with self.lock:
                if subscriber in self.subscribers:
                    message = self.subscribers[subscriber].get_nowait()
                    subscriber.sendall(message.encode())
                    print(f"Sent message to subscriber: {message}")
        except Exception as e:
            print(f"Error sending message: {e}")
            self._close_connection(subscriber)

    def _close_connection(self, client_socket):
        """关闭客户端连接"""
        with self.lock:
            if client_socket in self.subscribers:
                del self.subscribers[client_socket]
                print(f"订阅者断开连接，总订阅者数量: {len(self.subscribers)}")
        self.sel.unregister(client_socket)
        client_socket.close()

    def start(self):
        try:
            while True:
                events = self.sel.select(timeout=None)
                for key, _ in events:
                    callback = key.data
                    callback(key.fileobj)
        except KeyboardInterrupt:
            print("Shutting down broker.")
        finally:
            self.server.close()
            self.message_queue.put(None)  # 停止消息处理线程

if __name__ == "__main__":
    broker = MessageBroker(host="localhost", port=9999)
    broker.start()
