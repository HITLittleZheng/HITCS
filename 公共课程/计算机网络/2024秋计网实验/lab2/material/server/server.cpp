#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <windows.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#pragma comment(lib, "ws2_32.lib")
#pragma warning(disable : 4996)
#define SERVER_PORT 12340     // 端口号
#define SERVER_IP   "0.0.0.0" // IP 地址
using namespace std;
const int BUFFER_LENGTH = 1027; // 缓冲区大小，（以太网中 UDP 的数据帧中包长度应小于 1480 字节）
const int SEND_WIND_SIZE = 10;  // 发送窗口大小为 10，GBN 中应满足 W + 1 <= N（W 为发送窗口大小，N 为序列号个数）
// 本例取序列号 0...19 共 20 个
// 如果将窗口大小设为 1，则为停-等协议
const int SEQ_SIZE = 20; // 序列号的个数，从 0~19 共计 20 个
// 由于发送数据第一个字节如果值为 0，则数据会发送失败
// 因此接收端序列号为 1~20，与发送端一一对应
BOOL ack[SEQ_SIZE]; // ack缓存
int curSeq;         // 当前数据包的 seq
int curAck;         // 当前等待确认的 ack
int totalSeq;       // 收到的包的总数
int totalPacket;    // 需要发送的包总数
int waitSeq;
//************************************
// Method: getCurTime
// FullName: getCurTime
// Access: public
// Returns: void
// Qualifier: 获取当前系统时间，结果存入 ptime 中
// Parameter: char * ptime
//************************************
void getCurTime(char *ptime) {
    char buffer[128];
    memset(buffer, 0, sizeof(buffer));
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    sprintf_s(
        buffer, "%4d/%02d/%02d %02d:%02d:%02d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond);
    strcpy_s(ptime, sizeof(buffer), buffer);
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
    printf("Timer out error.\n");
    int index;
    for (int i = 0; i < (curSeq - curAck + SEQ_SIZE) % SEQ_SIZE; ++i) {
        index = (i + curAck) % SEQ_SIZE;
        ack[index] = TRUE;
    }
    totalSeq = totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE);
    curSeq = curAck;
}
//************************************
// Method: ackHandler
// FullName: ackHandler
// Access: public
// Returns: void
// Qualifier: 收到 ack，累积确认，取数据帧的第一个字节
// 由于发送数据时，第一个字节（序列号）为 0（ASCII）时发送失败，因此加一了，此处需要减一还原
// Parameter: char c
//************************************
void ackHandler(char c) {
    unsigned char index = (unsigned char)c - 1; // 序列号减一
    printf("Recv a ack of %d\n", index);
    if (curAck <= index) {
        for (int i = curAck; i <= index; ++i) {
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
        curAck = index + 1;
    }
}
//************************************
// Method: lossInLossRatio
// Access: public
// Returns: BOOL
// Qualifier: 模拟随机丢包，返回TRUE则执行丢包
// Parameter: float lossRatio
//************************************
BOOL lossInLossRatio(float lossRatio) {
    int lossBound = (int)(lossRatio * 100);
    int r = rand() % 100;
    if (r < lossBound) {
        return TRUE;
    }
    return FALSE;
}
// 主函数
int main() {
    // 加载套接字库（必须）
    WORD wVersionRequested;
    WSADATA wsaData;
    // 套接字加载时错误提示
    int err;
    // 版本 2.2
    wVersionRequested = MAKEWORD(2, 2);
    // 加载 dll 文件 Scoket 库
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0) {
        // 找不到 winsock.dll
        printf("WSAStartup failed with error: %d\n", err);
        return -1;
    }
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
        printf("Could not find a usable version of Winsock.dll\n");
        WSACleanup();
    } else {
        printf("The Winsock 2.2 dll was found okay\n");
    }
    SOCKET sockServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    // 设置套接字为非阻塞模式
    int iMode;              // 1：非阻塞，0：阻塞
    SOCKADDR_IN addrServer; // 服务器地址
    // addrServer.sin_addr.S_un.S_addr = inet_addr(SERVER_IP);
    addrServer.sin_addr.S_un.S_addr = htonl(INADDR_ANY); // 两者均可
    addrServer.sin_family = AF_INET;
    addrServer.sin_port = htons(SERVER_PORT);
    err = bind(sockServer, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
    if (err) {
        err = GetLastError();
        printf("Could not bind the port %d for socket.Error code is % d\n", SERVER_PORT, err);
        WSACleanup();
        return -1;
    }

    SOCKADDR_IN addrClient; // 客户端地址
    int length = sizeof(SOCKADDR);
    char buffer[BUFFER_LENGTH]; // 数据发送接收缓冲区
    ZeroMemory(buffer, sizeof(buffer));
    // 将测试数据读入内存
    int recvSize;
    int loct = 0; // 记录下载的文件的行数
    int waitCount = 0;
    float packetLossRatio = 0.2; // 默认包丢失率 0.2
    float ackLossRatio = 0.2;    // 默认 ACK 丢失率 0.2
    srand((unsigned)time(NULL));
    while (true) {
        // 非阻塞接收，若没有收到数据，返回值为-1
        recvSize = recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&addrClient), &length);
        // 删除buffer中的回车

        printf("recv from client: %s\n", buffer);
        if (strcmp(buffer, "-time\n") == 0) {
            getCurTime(buffer);
        } else if (strcmp(buffer, "-quit\n") == 0) {
            strcpy_s(buffer, strlen("Good bye!") + 1, "Good bye!");
        } else {
            char filename[100];
            char operation[10];
            char cmd[10];
            int ret;                // sscanf 返回值
            unsigned char u_code;   // 状态码
            unsigned short seq;     // 包的序列号
            unsigned short recvSeq; // 接收窗口大小为 1，已确认的序列号
            unsigned short waitSeq; // 等待的序列号
            unsigned short recvPacket;
            int sendack = 0;
            int stage = 0;
            ret = sscanf(buffer, "%s %f %f %s %s", &cmd, &packetLossRatio, &ackLossRatio, &operation, &filename);
            if (!strcmp(cmd, "gbn")) {
                if (!strcmp(operation, "download")) {
                    iMode = 1;
                    int flg = 1;
                    ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                    std::ifstream fin;
                    fin.open(filename, ios_base::in);
                    if (!fin.is_open()) {
                        printf("无法打开文件");
                        iMode = 0;
                        ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                        continue;
                    }
                    char buff[1024] = {0};
                    char data[1024 * 113];
                    loct = 0;
                    while (fin.getline(buff, sizeof(buff))) {
                        if (buff[0] == '0')
                            break;
                        memcpy(data + 1024 * loct, buff, 1024);
                        ++loct;
                    }
                    fin.close();
                    totalPacket = loct;
                    ZeroMemory(buffer, sizeof(buffer));
                    int recvSize;
                    waitCount = 0;
                    printf("Begain to test GBN protocol,please don't abort the process\n");
                    // 加入了一个握手阶段
                    // 首先服务器向客户端发送一个 205 大小的状态码（我自己定义的）表示服务器准备好了，可以发送数据
                    // 客户端收到 205 之后回复一个 200 大小的状态码，表示客户端准备好了，可以接收数据了
                    // 服务器收到 200 状态码之后，就开始使用 GBN 发送数据了
                    printf("Shake hands stage\n");
                    int stage = 0;
                    bool runFlag = true;
                    while (runFlag) {
                        switch (stage) {
                        case 0: // 发送 205 阶段
                            buffer[0] = 205;
                            sendto(
                                sockServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                            Sleep(100);
                            stage = 1;
                            break;
                        case 1: // 等待接收 200 阶段，没有收到则计数器+1，超时则放弃此次“连接”，等待从第一步开始
                            recvSize =
                                recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&addrClient), &length);
                            if (recvSize < 0) {
                                ++waitCount;
                                if (waitCount > 20) {
                                    runFlag = false;
                                    printf("Timeout error\n");
                                    break;
                                }
                                Sleep(500);
                                continue;
                            } else {
                                if ((unsigned char)buffer[0] == 200) {
                                    printf("Begin a file transfer\n");
                                    printf(
                                        "File size is %dB, each packet is 1024B and packet total num is % d\n",
                                        totalPacket * 1024,
                                        totalPacket);
                                    curSeq = 0;
                                    curAck = 0;
                                    totalSeq = 0;
                                    waitCount = 0;
                                    stage = 2;
                                    for (int i = 0; i < SEQ_SIZE; ++i) {
                                        ack[i] = TRUE;
                                    }
                                }
                            }
                            break;
                        case 2: // 数据传输阶段
                            if (seqIsAvailable() && totalSeq < loct) {
                                // 发送给客户端的序列号从 1 开始
                                buffer[0] = curSeq + 1;
                                if (totalSeq == loct - 1)
                                    buffer[1] = '0';
                                else
                                    buffer[1] = '1';
                                ack[curSeq] = FALSE;
                                // 数据发送的过程中应该判断是否传输完成
                                // 为简化过程此处并未实现
                                memcpy(&buffer[2], data + 1024 * totalSeq, 1024);
                                printf("send a packet with a seq of %d\n", curSeq);
                                sendto(sockServer, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                                ++curSeq;
                                curSeq %= SEQ_SIZE;
                                ++totalSeq;
                                Sleep(500);
                            }
                            // 等待 Ack，若没有收到，则返回值为-1，计数器+1
                            recvSize =
                                recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&addrClient), &length);
                            if (recvSize < 0) {
                                waitCount++;
                                // 20 次等待 ack 则超时重传
                                if (waitCount > 20) {
                                    timeoutHandler();
                                    waitCount = 0;
                                }
                            } else {
                                // 收到 ack
                                if (buffer[1] == '0') {
                                    flg = 0;
                                    break;
                                }
                                ackHandler(buffer[0]);
                                waitCount = 0;
                            }
                            Sleep(500);
                            break;
                        }
                        if (flg == 0) {
                            break;
                        }
                    }
                    if (flg == 0) {
                        printf("传输完成\n");
                        iMode = 0;
                        ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                        ZeroMemory(buffer, sizeof(buffer));
                        continue;
                    }
                } else if (!strcmp(operation, "upload")) {
                    char data[1024 * 113];
                    loct = 0;
                    int flg = 1;
                    BOOL b;
                    // gbn 0 0 download test.txt
                    while (true) {
                        // 等待 server 回复设置 UDP 为阻塞模式
                        recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&addrClient, &length);
                        switch (stage) {
                        case 0: // 等待握手阶段
                            u_code = (unsigned char)buffer[0];
                            if ((unsigned char)buffer[0] == 205) {
                                printf("Ready for file transmission\n");
                                buffer[0] = 200;
                                buffer[1] = '\0';
                                sendto(sockServer, buffer, 2, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                                stage = 1;
                                recvSeq = 0;
                                waitSeq = 1;
                                loct = 0;
                            }
                            break;
                        case 1: // 等待接收数据阶段
                            seq = (unsigned short)buffer[0];
                            b = lossInLossRatio(packetLossRatio);
                            if (b) {
                                printf("The packet with a seq of %d loss\n", seq - 1);
                                continue;
                            }
                            printf("recv a packet with a seq of %d\n", seq - 1);
                            // 如果是期待的包，正确接收，正常确认即可
                            if (!(waitSeq - seq)) {
                                if (buffer[1] == '0')
                                    flg = 0;
                                memcpy(data + 1024 * loct, buffer + 2, 1024);
                                ++loct;
                                ++waitSeq;
                                if (waitSeq == 21) {
                                    waitSeq = 1;
                                }
                                // 输出数据
                                // printf("%s\n",&buffer[1]);
                                buffer[0] = seq;
                                recvSeq = seq;
                                recvPacket = (unsigned short)buffer[1];
                                buffer[2] = '\0';
                            } else {
                                // 如果当前一个包都没有收到，则等待 Seq 为 1 的数据包，不是则不返回
                                // ACK（因为并没有上一个正确的 ACK）
                                if (!recvSeq) {
                                    continue;
                                }
                                buffer[0] = recvSeq;
                                buffer[1] = recvPacket;
                                buffer[2] = '\0';
                            }
                            b = lossInLossRatio(ackLossRatio);
                            if (b) {
                                printf("The ack of %d loss\n", (unsigned char)buffer[0] - 1);
                                continue;
                            }
                            sendto(sockServer, buffer, 3, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                            printf("send a ack of %d\n", (unsigned char)buffer[0] - 1);
                            break;
                        }
                        if (flg == 0) {
                            printf("接收完成\n");
                            break;
                        }
                        Sleep(500);
                    }
                    ofstream ofs;
                    ofs.open(filename, ios::out);
                    char buff[1300];
                    printf("%d", loct);
                    for (int i = 0; i < loct; ++i) {
                        memcpy(buff, data + 1024 * i, 1024);
                        ofs << buff << endl;
                    }
                    ofs.close();
                    if (flg == 0) {
                        ZeroMemory(buffer, sizeof(buffer));
                        continue;
                    }
                }
            } else if (!strcmp(cmd, "sr")) {
                if (!strcmp(operation, "download")) {
                    iMode = 1;
                    ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                    std::ifstream fin;
                    fin.open(filename, ios_base::in);
                    if (!fin.is_open()) {
                        printf("无法打开文件");
                        iMode = 0;
                        ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                        continue;
                    }
                    char buff[1024] = {0};
                    char data[1024 * 113];
                    loct = 0;
                    while (fin.getline(buff, sizeof(buff))) {
                        if (buff[0] == '0')
                            break;
                        memcpy(data + 1024 * loct, buff, 1024);
                        ++loct;
                    }
                    fin.close();
                    totalPacket = loct;
                    ZeroMemory(buffer, sizeof(buffer));
                    int recvSize;
                    int waitCounts[21] = {0};
                    waitCount = 0;
                    printf("Begain to test SR protocol,please don't abort the process\n");
                    // 加入了一个握手阶段
                    // 首先服务器向客户端发送一个 205 大小的状态码,表示服务器准备好了，可以发送数据
                    // 客户端收到 205 之后回复一个 200 大小的状态码，表示客户端准备好了，可以接收数据了
                    // 服务器收到 200 状态码之后，就开始使用 GBN 发送数据了
                    printf("Shake hands stage\n");
                    int stage = 0;
                    bool runFlag = true;
                    while (runFlag) {
                        switch (stage) {
                        case 0: // 发送 205 阶段
                            buffer[0] = 205;
                            sendto(
                                sockServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                            Sleep(100);
                            stage = 1;
                            break;
                        case 1: // 等待接收 200 阶段，没有收到则计数器+1，超时则放弃此次“连接”，等待从第一步开始
                            recvSize =
                                recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&addrClient), &length);
                            if (recvSize < 0) {
                                ++waitCount;
                                if (waitCount > 20) {
                                    runFlag = false;
                                    printf("Timeout error\n");
                                    break;
                                }
                                Sleep(500);
                                continue;
                            } else {
                                if ((unsigned char)buffer[0] == 200) {
                                    printf("Begin a file transfer\n");
                                    printf(
                                        "File size is %dB, each packet is 1024B and packet total num is % d\n",
                                        totalPacket * 1024,
                                        totalPacket);
                                    curSeq = 0;
                                    curAck = 0;
                                    totalSeq = 0;
                                    waitCount = 0;
                                    waitSeq = 0;
                                    stage = 2;
                                    for (int i = 0; i < SEQ_SIZE; ++i) {
                                        ack[i] = TRUE;
                                    }
                                }
                            }
                            break;
                        case 2:
                            if (seqIsAvailable() && totalSeq < loct) {
                                // 发送给客户端的序列号从 1 开始
                                buffer[0] = curSeq + 1;
                                if (totalSeq == loct - 1)
                                    buffer[1] = '0';
                                else
                                    buffer[1] = '1';
                                ack[curSeq] = FALSE;
                                // 数据发送的过程中应该判断是否传输完成
                                // 为简化过程此处并未实现
                                memcpy(&buffer[2], data + 1024 * totalSeq, 1024);
                                printf("send a packet with a seq of %d\n", curSeq);
                                sendto(sockServer, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                                ++curSeq;
                                curSeq %= SEQ_SIZE;
                                ++totalSeq;
                                Sleep(500);
                            }
                            // 等待 Ack，若没有收到，则返回值为-1，计数器+1
                            recvSize =
                                recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR *)&addrClient), &length);
                            if (recvSize >= 0) {
                                // 收到 ack
                                int i = (int)(buffer[0]) - 1;
                                ack[i] = TRUE;
                                waitCounts[i] = 0;
                                printf("Recv a ack of %d\n", i);
                                if (i == curAck) {
                                    if (curSeq < curAck) {
                                        for (; curAck < SEQ_SIZE;) {
                                            if (ack[curAck])
                                                ++curAck;
                                            else
                                                break;
                                        }
                                        if (curAck == SEQ_SIZE) {
                                            for (curAck = 0; curAck < curSeq;) {
                                                if (ack[curAck])
                                                    ++curAck;
                                                else
                                                    break;
                                            }
                                        }
                                    } else {
                                        for (; curAck < curSeq;) {
                                            if (ack[curAck])
                                                ++curAck;
                                            else
                                                break;
                                        }
                                    }
                                }
                                if (curAck == curSeq && totalSeq == loct)
                                    break;
                            }
                            int index;
                            for (int i = 0; i < (curSeq - curAck + SEQ_SIZE) % SEQ_SIZE; ++i) {
                                index = (i + curAck) % SEQ_SIZE;
                                if (!ack[index]) {
                                    ++waitCounts[index];
                                    if (waitCounts[index] > 20) {
                                        buffer[0] = index + 1;
                                        if (totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE) + i == loct - 1)
                                            buffer[1] = '0';
                                        else
                                            buffer[1] = '1';
                                        memcpy(
                                            &buffer[2],
                                            data + 1024 * (totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE) + i),
                                            1024);
                                        printf("send a packet with a seq of %d\n", index);
                                        sendto(
                                            sockServer,
                                            buffer,
                                            BUFFER_LENGTH,
                                            0,
                                            (SOCKADDR *)&addrClient,
                                            sizeof(SOCKADDR));
                                        waitCounts[index] = 0;
                                    }
                                }
                            }
                            Sleep(500);
                            break;
                        }
                        if (curAck == curSeq && totalSeq == loct)
                            break;
                    }
                    if (curAck == curSeq && totalSeq == loct) {
                        printf("传输完成\n");
                        iMode = 0;
                        ioctlsocket(sockServer, FIONBIO, (u_long FAR *)&iMode);
                        ZeroMemory(buffer, sizeof(buffer));
                        continue;
                    }
                } else if (!strcmp(operation, "upload")) {
                    // sr 0 0 download test.txt
                    char data[1024 * 113];
                    BOOL recvd[20] = {FALSE};
                    sendack = 0;
                    BOOL b;
                    while (true) {
                        // 等待 server 回复设置 UDP 为阻塞模式
                        recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&addrClient, &length);
                        switch (stage) {
                        case 0: // 等待握手阶段
                            u_code = (unsigned char)buffer[0];
                            if ((unsigned char)buffer[0] == 205) {
                                printf("Ready for file transmission\n");
                                buffer[0] = 200;
                                buffer[1] = '\0';
                                sendto(sockServer, buffer, 2, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                                stage = 1;
                                recvSeq = 0;
                                waitSeq = 0;
                                totalSeq = 0;
                                loct = -2;
                            }
                            break;
                        case 1: // 等待接收数据阶段
                            seq = (unsigned short)buffer[0];
                            // 随机法模拟包是否丢失
                            b = lossInLossRatio(packetLossRatio);
                            if (b) {
                                printf("The packet with a seq of %d loss\n", seq - 1);
                                continue;
                            }
                            printf("recv a packet with a seq of %d\n", seq - 1);
                            // 如果是期待的包，正确接收，正常确认即可
                            seq -= 1;
                            if (!(waitSeq - seq)) {
                                recvd[waitSeq] = TRUE;
                                memcpy(data + 1024 * totalSeq, buffer + 2, 1024);
                                if (buffer[1] == '0')
                                    loct = totalSeq;
                                int cnt = 10;
                                while (cnt--) {
                                    if (recvd[waitSeq]) {
                                        recvd[waitSeq] = FALSE;
                                        ++waitSeq;
                                        ++totalSeq;
                                        if (waitSeq == 20)
                                            waitSeq = 0;
                                    } else
                                        break;
                                }
                            } else {
                                int index = (seq + SEQ_SIZE - waitSeq) % SEQ_SIZE;
                                if (index < 10 && !recvd[seq]) {
                                    recvd[seq] = TRUE;
                                    memcpy(data + 1024 * (totalSeq + index), buffer + 2, 1024);
                                    if (buffer[1] == '0')
                                        loct = totalSeq + index;
                                }
                            }
                            buffer[0] = (char)(seq + 1);
                            buffer[2] = '\0';
                            b = lossInLossRatio(ackLossRatio);
                            if (b) {
                                printf("The ack of %d loss\n", (unsigned char)buffer[0] - 1);
                                continue;
                            }
                            ++sendack;
                            sendto(sockServer, buffer, 3, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                            printf("send a ack of %d\n", (unsigned char)buffer[0] - 1);
                            break;
                        }
                        if (sendack == loct + 1) {
                            printf("接收完成\n");
                            break;
                        }
                        Sleep(500);
                    }
                    char buff[1300];
                    ofstream ofs;
                    ofs.open(filename, ios::out);
                    for (int i = 0; i <= loct; ++i) {
                        memcpy(buff, data + 1024 * i, 1024);
                        ofs << buff << endl;
                    }
                    ofs.close();
                    if (sendack == loct + 1) {
                        ZeroMemory(buffer, sizeof(buffer));
                        continue;
                    }
                }
            }
        }
        // 进入 gbn 测试阶段
        // 首先 server（server 处于 0 状态）向 client 发送 205 状态码（server进入 1 状态）
        //     server 等待 client 回复 200 状态码，如果收到（server 进入 2 状态），则开始传输文件，否则延时等待直至超时\
				//在文件传输阶段，server 发送窗口大小设为
        sendto(sockServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
        Sleep(500);
    }
    // 关闭套接字，卸载库
    closesocket(sockServer);
    WSACleanup();
    return 0;
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧:
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
