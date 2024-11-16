# SR

## 注意事项

> 本实验借鉴了往届学长学姐的代码思想，并且结合了自身对代码的理解，进行了重新的编写，代码的主体逻辑一直，并且我认为状态码设计是一个比较优良的设计，所以未进行状态码的更改。
>
> 先下载后上传

## client和server之间的状态码

- client 使用两位数的状态码
- server使用三位数的状态码
- 请务必保证状态码小于255

## 代码中的一些变量

start ack指的是最早的还未接受的确认的ack

## TODO 1

### **代码片段解析**

这部分代码实现了客户端在**下载数据包时的处理逻辑**，主要包括：接收数据包、处理接收窗口、发送 ACK（确认包）。我们逐行分析该逻辑的具体作用。

---

### **代码的作用详解**

```cpp
if (recvSize > 0 && (unsigned char)buffer[0] == 200) {
    printf("开始下载\n");
    start = clock();
    seq = buffer[1];  // 获取收到的数据包的序列号

    printf("Recv： seq = %2d, data = %s\n", seq, buffer + 2);  // 打印接收信息
    printf("Send： ack = %2d\n", seq);  // 打印要发送的 ACK

    seq--;  // TODO: 为什么要减1
```

#### **1. 为什么 `seq--`？**

- **序列号在接收窗口中的索引**：在 **SR 协议**中，序列号通常是从 1 开始的，而接收窗口的数组索引从 0 开始。因此，需要将序列号减 1，以确保序列号与接收窗口的索引对齐。
- **示例**：
  如果收到的包序列号是 1，则将其对应到 `recvWindow[0]`。因此 `seq--` 的操作是为了正确定位数组中的位置。

---

```cpp
    recvWindow[seq].used = true;  // 标记该包已接收
    strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);  // 将包数据存入接收窗口
```

#### **2. 将数据存入接收窗口**

- **`recvWindow[seq].used = true`**：标记该包为已接收。
- **`strcpy_s`**：将数据内容复制到 `recvWindow[seq].buffer`。

---

```cpp
    if (ack == seq) {
        ack = Deliver(file, ack);  // 如果当前包的序列号与 ACK 对齐，则尝试按序交付
    }
```

#### **3. 调用 `Deliver` 函数按序交付**

- 如果接收到的包的序列号与当前的 `ack` 对齐，调用 `Deliver` 函数，**按序处理接收到的数据包**。

---

```cpp
    stage = 2;  // 更新状态，进入下一阶段

    buffer[0] = 11;  // 构造 ACK 包，设置类型为 11
    buffer[1] = seq + 1;  // 序列号加 1，作为确认号发送给服务器
    buffer[2] = 0;  // 结束符

    sendto(socketClient, buffer, strlen(buffer) + 1, 0, 
           (SOCKADDR*)&addrServer, sizeof(SOCKADDR));  // 发送 ACK 包
    continue;
```

#### **4. 发送 ACK 包**

- **`buffer[0] = 11`**：设置包的类型为 11（ACK）。
- **`buffer[1] = seq + 1`**：ACK 包的序列号加 1，表示确认收到对应的数据包。
- **发送 ACK 包**：使用 `sendto` 发送确认消息。

---

### **总结**

1. **接收数据包**：接收到了序列号为 `buffer[1]` 的数据包。
2. **索引对齐**：`seq--` 的操作是为了将序列号与接收窗口的数组索引对齐。
3. **数据存储**：将接收到的数据存入 `recvWindow`。
4. **按序交付**：如果接收到的数据包按序排列，调用 `Deliver` 函数交付数据。
5. **发送 ACK**：构造 ACK 包并发送给服务器。

这段代码确保了 **SR 协议**中乱序接收的数据包能被正确存储，并在接收窗口中按序交付，同时通过发送 ACK 包通知服务器当前的接收状态。
