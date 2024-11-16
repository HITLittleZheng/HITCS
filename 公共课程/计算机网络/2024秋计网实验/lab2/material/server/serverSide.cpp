#include <WinSock2.h> // Windows下的Socket编程库，提供网络通信的基础API。
#include <stdlib.h>   // 标准库，包含基本的C函数，如内存管理、进程控制等。
#include <time.h>     // 时间相关的库，提供处理和获取系统时间的函数。
#include <windows.h>  // Windows API，提供与操作系统交互的功能，如系统时间、进程控制等。

#include <cstdio>  // 标准输入输出库，提供基本的输入输出函数，如`printf`、`scanf`等。
#include <fstream> // 文件输入输出流库，提供读写文件的功能。
#include <sstream> // 字符串流库，提供格式化和解析字符串的功能。

#pragma comment(lib, "ws2_32.lib") // 链接Winsock库，该库提供Windows下的网络编程功能。
#pragma warning(disable : 4996)    // 禁用4996号警告，通常用于禁止旧版函数的安全性警告（如`scanf`、`strcpy`）。

#define SERVER_PORT 12340     // 定义服务器端口号，服务器将监听该端口上的请求。
#define SERVER_IP   "0.0.0.0" // 定义服务器IP地址，"0.0.0.0"表示监听所有网络接口上的请求。
const int SEQ_SIZE = 20;      // 序列号的个数，从 0~19 共计 20 个
// 由于发送数据第一个字节如果值为 0，则数据会发送失败
// 因此接收端序列号为 1~20，与发送端一一对应
BOOL ack[SEQ_SIZE]; // 收到 ack 情况，对应 0~19 的 ack
                    // 为true的时候代表当前位置可用
int curSeq;         // 当前数据包的 seq
int curAck;         // 当前等待确认的 ack
int totalSeq;       // 收到的包的总数
int totalPacket;    // 需要发送的包总数
using namespace std;
// UDP数据帧有大小限制？
const int BUFFER_LENGTH = 1027; // 缓冲区大小，（以太网中 UDP 的数据帧中包长度应小于 1480 字节）
const int SEND_WIND_SIZE = 10;  // 发送窗口大小为 10，GBN 中应满足 W + 1 <= N（W
                                // 为发送窗口大小，N 为序列号个数）

bool timeoutTriggered = false;
int initializeNetwork();
SOCKET
bindSocket();
void cleanup(SOCKET *serverSocket);
void getCurTime(char *buffer);
void processClientRequest(SOCKET serverSocket);
int main();
void handleProtocalRequest(SOCKET serverSocket, SOCKADDR_IN clientAddr, char *buffer, int length);
void handleGBNDownload(
    SOCKET serverSocket,
    SOCKADDR_IN clientAddr,
    const char *filename,
    float packetLossRatio,
    float ackLossRatio,
    int length);
void handleGBNUpload(
    SOCKET serverSocket,
    SOCKADDR_IN clientAddr,
    const char *filename,
    float packetLossRatio,
    float ackLossRatio,
    int length);
void handleSRDownload(
    SOCKET serverSocket,
    SOCKADDR_IN clientAddr,
    const char *filename,
    float packetLossRatio,
    float ackLossRatio,
    int length);
void handleSRUpload(
    SOCKET serverSocket,
    SOCKADDR_IN clientAddr,
    const char *filename,
    float packetLossRatio,
    float ackLossRatio,
    int length);
void removeEnter(char *buffer);
bool seqIsAvailable();
void timeoutHandler();

// 加载winsock2库
int initializeNetwork() {
    WORD wVersionRequested = MAKEWORD(2, 2);           // 请求的Winsock库版本
    WSADATA wsaData;                                   // 用于存储Winsock库的信息
    int err = WSAStartup(wVersionRequested, &wsaData); // 加载Winsock库
    if (err != 0) {                                    // 加载失败
        printf("WSAStartup failed with error: %d\n", err);
        return -1;
    }

    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) { // 检查加载的Winsock库版本
        printf("Could not find a usable version of Winsock.dll\n");
        WSACleanup();
        return -1;
    }
    printf("The Winsock 2.2 dll was found okay\n");
    return 0;
}

//************************************
// Method: seqIsAvailable
// FullName: seqIsAvailable
// Access: public
// Returns: bool
// Qualifier: 当前序列号 curSeq 是否可用
//************************************
bool seqIsAvailable() {
    int step;
    step = curSeq - curAck;
    step = step >= 0 ? step : step + SEQ_SIZE;
    // 序列号是否在当前发送窗口之内
    if (step >= SEND_WIND_SIZE) {
        return false;
    }
    if (ack[curSeq]) {
        return true;
    }
    return false;
}

//************************************
// Method: timeoutHandler
// FullName: timeoutHandler
// Access: public
// Returns: void
// Qualifier: 超时重传处理函数，滑动窗口内的数据帧都要重传
//************************************
void timeoutHandler() {
    printf("ACK 超时\n");
    // 仅更新标志位，由主循环处理重传逻辑
    timeoutTriggered = true;
}


//************************************
// Method: ackHandler
// FullName: ackHandler
// Access: public
// Returns: void
// Qualifier: 收到 ack，累积确认，取数据帧的第一个字节
// 由于发送数据时，第一个字节（序列号）为
// 0（ASCII）时发送失败，因此加一了，此处需要减一还原 Parameter: char c
//************************************
void ackHandler(char c) {
    unsigned char index = (unsigned char)c - 1; // 序列号减一， 因为客户端发送的时候加了1
    printf("Recv a ack of %d\n", index);
    if (curAck <= index) {
        for (int i = curAck; i <= index; ++i) {
            // 当前确认接受到的包，到之前确认的包都可以接受
            ack[i] = TRUE;
        }
        curAck = (index + 1) % SEQ_SIZE;
    } else {
        // ack 超过了最大值，回到了 curAck 的左边
        for (int i = curAck; i < SEQ_SIZE; ++i) {
            ack[i] = TRUE;
        }
        for (int i = 0; i <= index; ++i) {
            ack[i] = TRUE;
        }
        curAck = index + 1; // 更新当前确认的包的序号(下一个包)
    }
}

// 初始化一个socket  绑定本地的ip(可访问的ip吗？)和端口
// 所有的udp连接使用共同的socket
SOCKET bindSocket() {
    // 创建的是一个数据报（Datagram）套接字。 指定为UDP协议
    SOCKET serverSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); // 创建一个UDP套接字
    // 阻塞与非阻塞应当在上传或者下载的时候进行具体的设置
    // int iMode = 1; // 1：非阻塞，0：阻塞
    // ioctlsocket(serverSocket, FIONBIO, (u_long FAR *)&iMode);  // 非阻塞设置
    if (serverSocket == INVALID_SOCKET) { // 创建失败
        printf("Create socket failed with error: %d\n", WSAGetLastError());
        return INVALID_SOCKET;
    }

    sockaddr_in serverAddr;          // 服务器地址
    serverAddr.sin_family = AF_INET; // 设置地址族为IPv4
    // 允许来自任意ip的访问
    serverAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY); // 设置服务器地址为本地所有IP
    serverAddr.sin_port = htons(SERVER_PORT);            // 设置端口号12340
    // (SOCKADDR *)& 从sockaddr_in类型转换为sockaddr类型
    int err = bind(serverSocket, (SOCKADDR *)&serverAddr,
                   sizeof(SOCKADDR)); // 绑定套接字
    if (err == SOCKET_ERROR) {        // 绑定失败
        err = GetLastError();
        printf("Could not bind the port %d for socket.Error code is % d\n", SERVER_PORT, err);
        closesocket(serverSocket); // 关闭套接字
        return INVALID_SOCKET;
    }
    return serverSocket;
}

void cleanup(SOCKET *serverSocket) {
    closesocket(*serverSocket);
    WSACleanup();
}

void getCurTime(char *buffer) {
    char *time;
    time = (char *)malloc(128);
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    sprintf_s(time, 128, "%4d/%02d/%02d %02d:%02d:%02d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond);
    strcpy_s(buffer, 128, time);
    free(time);
}

void processClientRequest(SOCKET serverSocket) {
    SOCKADDR_IN clientAddr; // 客户端地址
    int length = sizeof(SOCKADDR);
    char buffer[BUFFER_LENGTH]; // 接收缓冲区
    ZeroMemory(buffer, sizeof(buffer));
    int recvSize;

    while (true) {
        int flag = 0;
        // 接收来自客户端的数据
        // recvfrom() 从绑定的套接字接收数据，并将数据存储到缓冲区中
        // 不断接受请求，由于udp的无连接特性，不需要accept
        // 并且UDP的Socket是复用的。
        // 数据报之间的处理应当不是顺序的，而是并发的。但是得在下面写多线程，如果不写多线程的话
        // 数据报之间应当是串行的
        recvSize = recvfrom(serverSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&clientAddr, &length);
        if (recvSize <= 0) {
            Sleep(100);
            continue;
        }
        printf("Received a packet from %s:%d\n", inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));
        printf("Data: %u\n", buffer[0]);
        if (!strcmp(buffer, "-quit\n")) {
            // 执行退出的逻辑
            strcpy_s(buffer, strlen("Good bye!") + 1, "Good bye!");
        } else if (!strcmp(buffer, "-time\n")) {
            // 执行获取时间的逻辑
            getCurTime(buffer);
        } else {
            // 根据具体的协议进行处理，GBN SR
            // 没有传递length参数
            // length = sizeof(SOCKADDR);
            flag = 1;
            handleProtocalRequest(serverSocket, clientAddr, buffer, length);
        }
        if (!flag) {
            sendto(serverSocket, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&clientAddr, sizeof(SOCKADDR));
        }
    }
}

// 删除buffer中的回车
void removeEnter(char *buffer) {
    for (int i = 0; buffer[i] != '\0'; i++) {
        if (buffer[i] == '\r' || buffer[i] == '\n') {
            buffer[i] = '\0';
        }
    }
}

void handleProtocalRequest(SOCKET serverSocket, SOCKADDR_IN clientAddr, char *buffer, int length) {

    char cmd[10];
    char operation[10];
    char filename[100];
    float packetLossRatio = 0.2; // 默认包丢失率
    float ackLossRatio = 0.2;    // 默认ACK丢失率
    // gbn 0.2 0.2 download testdownload.txt
    int result = sscanf(buffer, "%s %f %f %s %s", &cmd, &packetLossRatio, &ackLossRatio, &operation, &filename);
    if (result != 5) {
        return;
    }
    if (strcmp(cmd, "gbn") == 0) {
        if (strcmp(operation, "download") == 0) {
            handleGBNDownload(serverSocket, clientAddr, filename, packetLossRatio, ackLossRatio, length);
        } else if (strcmp(operation, "upload") == 0) {
            handleGBNUpload(serverSocket, clientAddr, filename, packetLossRatio, ackLossRatio, length);
        }
    } else if (strcmp(cmd, "sr") == 0) {
        if (strcmp(operation, "download") == 0) {
            handleSRDownload(serverSocket, clientAddr, filename, packetLossRatio, ackLossRatio, length);
        } else if (strcmp(operation, "upload") == 0) {
            handleSRUpload(serverSocket, clientAddr, filename, packetLossRatio, ackLossRatio, length);
        }
    }
}

// 设置serverSocket为非阻塞模式
void setNonBlocking(SOCKET serverSocket) {
    int iMode = 1;
    ioctlsocket(serverSocket, FIONBIO, (u_long FAR *)&iMode);
}

// 设置serverSocket为阻塞模式
void setBlocking(SOCKET serverSocket) {
    int iMode = 0;
    ioctlsocket(serverSocket, FIONBIO, (u_long FAR *)&iMode);
}

// 根据文件名读取文件到内存中，直到EOF标记为止
int readFileToMemory(const char *filename, char *data, int &loct, int &totalPacket) {
    std::ifstream fin;
    fin.open(filename, std::ios_base::in | std::ios::binary); // 以二进制模式打开文件

    if (!fin.is_open()) {
        printf("无法打开文件\n");
        return -1;
    }

    char buff[1024] = {0}; // 每次读取最多 1024 字节
    loct = 0;

    // 读取文件内容直到 EOF
    while (!fin.eof()) {
        fin.read(buff, sizeof(buff));             // 尝试读取 1024 字节
        std::streamsize bytesRead = fin.gcount(); // 实际读取的字节数

        if (bytesRead <= 0) {
            break; // 结束读取
        }

        // 将读取的数据复制到目标内存中
        memcpy(data + 1024 * loct, buff, bytesRead);
        ++loct; // 增加数据包计数

        memset(buff, 0, sizeof(buff)); // 清空缓冲区，避免残留数据影响下一次读取
    }

    totalPacket = loct; // 更新总包数
    fin.close();
    printf("文件读取完成，共读取 %d 个数据包\n", loct);
    // printf("文件内容：\n%s\n", data);
    return 0; // 成功返回
}

void handleGBNDownload(SOCKET serverSocket, SOCKADDR_IN clientAddr, const char *filename, float packetLossRatio, float ackLossRatio, int length) {
    char data[1024 * 113];
    char buffer[BUFFER_LENGTH];
    int loct = 0;
    setNonBlocking(serverSocket);                                          // 设置为非阻塞模式
    int readSuccess = readFileToMemory(filename, data, loct, totalPacket); // 读取文件到内存中
    if (readSuccess == -1) {
        setBlocking(serverSocket);
    }
    ZeroMemory(buffer, sizeof(buffer));
    int recvSize;
    int waitCount = 0;   // 等待的序列号
    int stage = 0;       // 状态机阶段
    bool runtime = true; // 运行标志
    bool transferIsDone = false;

    curSeq = 0;   // 初始化序列号
    curAck = 0;   // 初始化 ACK
    totalSeq = 0; // 总的发送包数
    // 进行握手
    while (runtime) {
        switch (stage) {
        case 0:
            printf("尝试与客户端进行握手\n");
            buffer[0] = (unsigned char)205;
            sendto(serverSocket, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&clientAddr, sizeof(SOCKADDR));
            Sleep(100);
            stage = 1;
            break; // break 语句用于跳出 switch 语句
        case 1:
            recvSize = recvfrom(serverSocket, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&clientAddr), &length);
            if (recvSize <= 0) {
                ++waitCount;
                if (waitCount > 20) {
                    runtime = false;
                    printf("握手阶段超时\n");
                    break; // 跳出switch 暂停是因为runtime = false
                }
                Sleep(100);
                continue;
            }
            if ((unsigned char)buffer[0] == 200) {
                printf("接受到客户端200状态码, 开始进行文件传输\n");
                stage = 2;
                curSeq = 0;   // seq是0 窗口长度为10
                curAck = 0;   // Ack长度为20 0~19
                totalSeq = 0; // 总的seq数量，这里等于包的数量
                waitCount = 0;
                // 当前所有的ack都是true 均可用
                for (int i = 0; i < SEQ_SIZE; ++i) {
                    ack[i] = TRUE;
                }
            }
            break;                                      
        case 2:                                        // 数据传输阶段
            if (seqIsAvailable() && totalSeq < loct) { // 确保包数未超出总包数
                buffer[0] = curSeq + 1;                // 序列号从 1 开始

                // 检查是否为最后一个包
                buffer[1] = (totalSeq == loct - 1) ? '0' : '1';
                ack[curSeq] = FALSE; // 标记为未确认

                // 复制数据块
                memcpy(&buffer[2], data + 1024 * totalSeq, 1024);
                printf("发送包，序列号：%d\n", curSeq);

                // 发送数据包
                sendto(serverSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&clientAddr, sizeof(SOCKADDR));

                curSeq = (curSeq + 1) % SEQ_SIZE; // 更新序列号
                ++totalSeq;                       // 更新已发送包数
                Sleep(100);                       // 控制发送速度
            } else if (totalSeq >= loct) {        // 防止发送超过总包数
                transferIsDone = true;
            }

            // 等待 ACK
            recvSize = recvfrom(serverSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&clientAddr, &length);
            if (recvSize < 0) {
                ++waitCount;
                if (waitCount > 20) { // 超时则触发重传逻辑
                    printf("ACK 超时，准备重传未确认的包\n");
                    timeoutTriggered = true;
                    waitCount = 0;
                }
            } else {
                if (buffer[1] == '0') { // 检测结束标志
                    transferIsDone = true;
                    break;
                }
                ackHandler(buffer[0]); // 处理 ACK
                waitCount = 0;
            }

            // 在主循环内处理超时后的重传逻辑
            if (timeoutTriggered) {
                printf("重传未确认的包\n");
                for (int i = 0; i < SEND_WIND_SIZE; ++i) {
                    int index = (curAck + i) % SEQ_SIZE; // 窗口内的包序列号

                    if (!ack[index] && totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE) + i < loct) {
                        buffer[0] = index + 1;                       // 更新包的序列号
                        buffer[1] = (index == loct - 1) ? '0' : '1'; // 标记最后一个包

                        memcpy(&buffer[2], data + 1024 * index, 1024); // 复制数据块
                        printf("重传包，序列号：%d\n", index);

                        sendto(serverSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&clientAddr, sizeof(SOCKADDR));
                    }
                }
                timeoutTriggered = false; // 重置超时标志
            }

            Sleep(100); // 控制传输速度
        }
        if (transferIsDone) {
            printf("传输完成\n");
            runtime = false;
            setBlocking(serverSocket);
            ZeroMemory(buffer, sizeof(buffer));
            break;
        }
    }
}

void handleGBNUpload(SOCKET serverSocket, SOCKADDR_IN clientAddr, const char *filename, float packetLossRatio, float ackLossRatio, int length) {}

void handleSRDownload(SOCKET serverSocket, SOCKADDR_IN clientAddr, const char *filename, float packetLossRatio, float ackLossRatio, int length) {}

void handleSRUpload(SOCKET serverSocket, SOCKADDR_IN clientAddr, const char *filename, float packetLossRatio, float ackLossRatio, int length) {}

int main() {
    // initialize network
    if (initializeNetwork() == -1) {
        return -1;
    }
    // bind socket
    SOCKET serverSocket = bindSocket();
    if (serverSocket == INVALID_SOCKET) {
        WSACleanup();
        return -1;
    }

    processClientRequest(serverSocket);

    cleanup(&serverSocket);
    return 0;
}