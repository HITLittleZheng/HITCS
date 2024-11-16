import socket
import threading

# 实现消息队列类
class MessageBroker:
    def __init__(self, host="localhost", port=9999):
        self.host = host
        self.port = port
        self.subscribers = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        print(f"Message Broker running on {self.host}:{self.port}")

    def start(self):
        try:
            while True:
                client_socket, addr = self.server.accept()
                print(f"Connection from {addr}")
                threading.Thread(
                    target=self.handle_client, args=(client_socket,), daemon=True
                ).start()
        except KeyboardInterrupt:
            print("Shutting down broker.")
            self.server.close()

    def handle_client(self, client_socket):
        buffer = ""  

        try:
            while "\n" not in buffer:
                data = client_socket.recv(1024).decode()
                if not data:
                    print("客户端连接中断")
                    return  # 没有收到数据则直接退出
                buffer += data

            role, buffer = buffer.split("\n", 1)  
            role = role.strip()  
            print(f"客户端角色: {role}")

            if role == "PUBLISHER":
                self.handle_publisher(client_socket, buffer)  # 传入剩余的缓冲数据
            elif role == "SUBSCRIBER":
                self.subscribers.append(client_socket)
                print(f"新增订阅者，总订阅者数量: {len(self.subscribers)}")
                while True:
                    # Keep the subscriber connection alive
                    data = client_socket.recv(1024)
                    if not data:
                        break
            else:
                print("未知角色，关闭连接")
                client_socket.close()

        except Exception as e:
            print(f"处理客户端时出错: {e}")

        finally:
            if client_socket in self.subscribers:
                self.subscribers.remove(client_socket)
                print(f"订阅者断开连接，总订阅者数量: {len(self.subscribers)}")
            client_socket.close()

    def handle_publisher(self, client_socket, buffer):
        try:
            
            # message = client_socket.recv(4096).decode()
            message = buffer
            if not message:
                print("Publisher connection closed.")
            print(f"Received message: {message}")
            self.broadcast(message)
        except Exception as e:
            print(f"Error handling publisher: {e}")
        finally:
            client_socket.close()

    def broadcast(self, message):
        to_remove = []
        for subscriber in self.subscribers:
            try:
                print(f"Sending message to subscriber: {message}")
                subscriber.sendall(message.encode())
            except:
                to_remove.append(subscriber)
        for subscriber in to_remove:
            self.subscribers.remove(subscriber)
            subscriber.close()
            print(f"Removed a subscriber. Total subscribers: {len(self.subscribers)}")


if __name__ == "__main__":
    broker = MessageBroker(host="localhost", port=9999)
    broker.start()
