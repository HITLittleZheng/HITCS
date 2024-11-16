#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")
#define SERVER_PORT   12345       // 接收数据的端口号
#define SERVER_IP     "127.0.0.1" // 服务器的 IP 地址
#define BUFFER_SIZE   1024        // 缓冲区大小
#define SEQ_SIZE      20          // 序列号个数
#define SWIN_SIZE     10          // 发送窗口大小
#define RECV_WIN_SIZE 10          // 接收窗口大小
#define LOSS_RATE     0.2         // 丢包率
using namespace std;

char cmdBuffer[50];
char buffer[BUFFER_SIZE];
char cmd[10];
char fileName[40];
char filePath[50];
char file[1024 * 1024];
int len = sizeof(SOCKADDR);
int recvSize;

int Deliver(char* file, int ack);
int Read(ifstream& infile, char* buffer);
int Send(ifstream& infile, int seq, SOCKET socket, SOCKADDR* addr);
int MoveSendWindow(int seq);

void Tips() {
    printf("======================================================================\n");
    printf("up <filename> 上传\n");
    printf("dl <filename> 下载\n");
    printf("quit 退出\n");
    printf("time 当前时间\n");
    printf("======================================================================\n");
}

struct RecvWindow {
    bool used;
    char buffer[BUFFER_SIZE];
    RecvWindow() {
        used = false; // 当前位置没有被使用
        ZeroMemory(buffer, sizeof(buffer));
    }
} recvWindow[SEQ_SIZE];
struct SendWindow {
    clock_t start;
    char buffer[BUFFER_SIZE];
    SendWindow() {
        start = 0;
        ZeroMemory(buffer, sizeof(buffer));
    }
} sendWindow[SEQ_SIZE];

int initailizeNetwork() {
    // 加载套接字库
    WORD wVersionRequested;
    WSADATA wsaData;
    // 版本 2.2
    wVersionRequested = MAKEWORD(2, 2);
    int err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0) {
        return -1;
    }
    if (LOBYTE(wsaData.wVersion) != LOBYTE(wVersionRequested) || HIBYTE(wsaData.wVersion) != HIBYTE(wVersionRequested)) {
        WSACleanup();
        return -1;
    }
    return 0;
}

// 设置非阻塞模式
int setNonBlocking(SOCKET socket) {
    u_long iMode = 1;
    return ioctlsocket(socket, FIONBIO, &iMode);
}

// 设置阻塞模式
int setBlocking(SOCKET socket) {
    u_long iMode = 0;
    return ioctlsocket(socket, FIONBIO, &iMode);
}

void InitializeVariables(int& seq, int& ack, int& stage, clock_t& start, clock_t& now, char* filePath, char* file) {
    seq = 0;
    ack = 0;
    stage = 0;
    start = 0;
    now = 0;
    ZeroMemory(filePath, sizeof(filePath));
    ZeroMemory(file, sizeof(file));
    // 重置接收窗口
    for (int i = 0; i < SEQ_SIZE; ++i) {
        recvWindow[i].used = false;
        ZeroMemory(recvWindow[i].buffer, sizeof(recvWindow[i].buffer));
    }

    // 重置发送窗口
    for (int i = 0; i < SEQ_SIZE; ++i) {
        sendWindow[i].start = 0;
        ZeroMemory(sendWindow[i].buffer, sizeof(sendWindow[i].buffer));
    }
}

int main() {
    int err = initailizeNetwork();
    if (err == -1) {
        return -1;
    }
    // 创建客户端套接字
    SOCKET socketClient = socket(AF_INET, SOCK_DGRAM, 0);
    // 设置为非阻塞模式
    setNonBlocking(socketClient);

    // 设置服务器地址
    SOCKADDR_IN addrServer;
    inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
    addrServer.sin_family = AF_INET;
    addrServer.sin_port = htons(SERVER_PORT);

    // 一些初始化
    srand((unsigned)time(NULL));
    int stage = 0;
    clock_t start;
    clock_t now;
    int seq;
    int ack;

    Tips();

    while (true) {
        gets_s(cmdBuffer, 50);
        sscanf_s(cmdBuffer, "%s%s", cmd, sizeof(cmd) - 1, fileName, sizeof(fileName) - 1);
        if (!strcmp(cmd, "dl")) {
            InitializeVariables(seq, ack, stage, start, now, filePath, file);
            printf("Download： %s\n", fileName);
            strcpy_s(filePath, "./");
            strcat_s(filePath, fileName);
            ofstream outfile(filePath);
            start = clock();
            ack = 0;
            stage = 0;
            // download中 sendWindow只在
            sendWindow[0].buffer[0] = 10; // 设置buffer 0为10 准备进行握手
            strcpy_s(sendWindow[0].buffer + 1, strlen(cmdBuffer) + 1, cmdBuffer);
            sendWindow[0].start = start - 1000L; // 直接进入初始牵手的初始化ATTENTION逻辑
            while (true) {
                recvSize = recvfrom(socketClient, buffer, BUFFER_SIZE, 0, (SOCKADDR*)&addrServer, &len);

                if ((float)rand() / RAND_MAX < LOSS_RATE) {
                    recvSize = 0;
                    buffer[0] = 0; // 约等于丢包
                }
                switch (stage) {
                case 0:
                    if (recvSize > 0 && buffer[0] == 100) {
                        if (!strcmp(buffer + 1, "OK")) {
                            printf("准备下载\n");
                            start = clock();
                            stage = 1;
                            sendWindow[0].buffer[0] = 10;
                            strcpy_s(sendWindow[0].buffer + 1, 3, "OK");
                            // 进入case 1 中的ATTENTION 2 逻辑。
                            sendWindow[0].start = start - 1000L;
                            continue;
                        } else if (!strcmp(buffer + 1, "NO")) {
                            stage = -1;
                            break;
                        }
                    }
                    now = clock();
                    // 直接进入的逻辑
                    // ATTENTION 1
                    if (now - sendWindow[0].start >= 1000L) {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 1:
                    // 丢包
                    if ((recvSize > 0) && ((float)rand() / RAND_MAX < LOSS_RATE)) {
                        printf("出现丢包\n");
                        recvSize = 0;
                        buffer[0] = 0; // 下面的代码设计具体的对recv的buffer的处理
                    }
                    if (recvSize > 0 && (unsigned char)buffer[0] == 200) {
                        printf("开始下载\n");
                        start = clock();
                        seq = buffer[1];
                        printf("Recv： seq = %2d, data = %s\n", seq, buffer + 2);
                        printf("Send： ack = %2d\n", seq);
                        seq--; // TODO: 1. 为什么要减1 @MarchPhantasia
                        // 请看readme
                        recvWindow[seq].used = true; // send ack 就是确认了，自然也需要设置为true
                        strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                        if (ack == seq) { // ack 用的是数组索引的形式
                                          // 这里拼写一次之后就在case2进行剩下的拼写了
                            ack = Deliver(file, ack);
                        }
                        stage = 2;
                        buffer[0] = 11;
                        buffer[1] = seq + 1;
                        buffer[2] = 0;
                        sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                        continue;
                    }
                    now = clock();
                    // ATTENTION 2 逻辑，从case 1 设置完stage而来
                    if (now - sendWindow[0].start >= 1000L) {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 2:
                    // TODO: 2.
                    if ((recvSize > 0) && ((float)rand() / RAND_MAX < LOSS_RATE)) {
                        printf("出现丢包\n");
                        recvSize = 0;
                        buffer[0] = 0;
                    }
                    if (recvSize > 0) {
                        if ((unsigned char)buffer[0] == 200) {
                            seq = buffer[1];
                            int temp = seq - 1 - ack; // 当前接收到的包的序号减去已经确认的包的序号
                            if (temp < 0) {
                                temp += SEQ_SIZE;
                            }
                            start = clock();
                            seq--;
                            // 发送、接受窗口大小一致
                            if (temp < RECV_WIN_SIZE) {
                                if (!recvWindow[seq].used) {
                                    recvWindow[seq].used = true;
                                    strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                                }

                                if (ack == seq) {
                                    ack = Deliver(file, ack);
                                }
                            }
                            printf("Recv： seq = %2d, data = %s\n", seq + 1, buffer + 2);
                            printf("Send： ack = %2d, Start ack = %2d\n", seq + 1, ack + 1);
                            buffer[0] = 11;
                            buffer[1] = seq + 1;
                            buffer[2] = 0;
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                        } else if (buffer[0] == 100 && !strcmp(buffer + 1, "Finish")) {
                            stage = 3;
                            outfile.write(file, strlen(file));
                            buffer[0] = 10;
                            strcpy_s(buffer + 1, 3, "OK");
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                            continue;
                        }
                    }
                    break;
                default:
                    break;
                }
                if (stage == 3) {
                    printf("下载完成\n");
                    outfile.close();
                    break;
                }
                if (clock() - start >= 5000L) {
                    printf("传输超时\n");
                    outfile.close();
                    break;
                }
                if (recvSize <= 0) {
                    Sleep(20);
                }
            }
            outfile.close();
        } else if (!strcmp(cmd, "up")) { // 上传
            InitializeVariables(seq, ack, stage, start, now, filePath, file);
            strcpy_s(filePath, "./");
            strcat_s(filePath, fileName);
            ifstream infile(filePath); // 打开文件
            start = clock();           // 返回自程序启动以来经过的时钟周期数。单位是毫秒
            seq = 0;
            stage = 0; // 发送状态

            sendWindow[0].buffer[0] = 10;
            strcpy_s(sendWindow[0].buffer + 1, strlen(cmdBuffer) + 1, cmdBuffer);
            // 直接触发case0的<超时发送>逻辑
            sendWindow[0].start = start - 1000L;
            while (true) {
                recvSize = recvfrom(socketClient, buffer, BUFFER_SIZE, 0, (SOCKADDR*)&addrServer, &len);
                switch (stage) {
                case 0:
                    if (recvSize > 0 && buffer[0] == 100) {
                        if (!strcmp(buffer + 1, "OK")) {
                            printf("开始上传\n");
                            start = clock();
                            stage = 1;
                            sendWindow[0].start = 0L;
                            continue;
                        } else if (!strcmp(buffer + 1, "NO")) {
                            stage = -1;
                            break;
                        }
                    }
                    now = clock();
                    // 超时发送 上面代码直接触发的逻辑。
                    // 握手阶段的超时重传
                    if (now - sendWindow[0].start >= 1000L) {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 1:
                    if (recvSize > 0 && buffer[0] == 101) {
                        start = clock();
                        ack = buffer[1];
                        ack--;
                        sendWindow[ack].start = -1L; // 已确认此ack
                        if (ack == seq) {
                            seq = MoveSendWindow(seq);
                        }
                        printf("Recv： ack = %2d, Current seq = %2d\n", ack + 1, seq + 1);
                    }
                    if (!Send(infile, seq, socketClient, (SOCKADDR*)&addrServer)) {
                        printf("上传成功\n");
                        stage = 2;
                        start = clock();
                        sendWindow[0].buffer[0] = 10;
                        strcpy_s(sendWindow[0].buffer + 1, 7, "Finish");
                        sendWindow[0].start = start - 1000L;
                        continue;
                    }
                    break;
                case 2:
                    if (recvSize > 0 && buffer[0] == 100) {
                        if (!strcmp(buffer + 1, "OK")) {
                            buffer[0] = 10;
                            strcpy_s(buffer + 1, 3, "OK");
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                            stage = 3;
                            break;
                        }
                    }
                    now = clock();
                    if (now - sendWindow[0].start >= 1000L) {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
                    }
                default:
                    break;
                }
                if (stage == 3) {
                    // printf("上传成功\n");
                    infile.close();
                    break;
                }
                if (clock() - start >= 5000L) {
                    printf("Timeout\n");
                    infile.close();
                    break;
                }
                if (recvSize <= 0) {
                    Sleep(200);
                }
            }
            infile.close();
        } else if (!strcmp(cmd, "time")) {
            setBlocking(socketClient);
            char tempBuffer[BUFFER_SIZE];
            ZeroMemory(tempBuffer, sizeof(tempBuffer));
            tempBuffer[0] = 10;
            strcpy_s(tempBuffer + 1, 5, "time");
            sendto(socketClient, tempBuffer, strlen(tempBuffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
            int ret = recvfrom(socketClient, tempBuffer, BUFFER_SIZE, 0, (SOCKADDR*)&addrServer, &len);
            if (ret > 0) {
                printf("当前时间: %s\n", tempBuffer + 1);
            }
            setNonBlocking(socketClient);
        } else if (!strcmp(cmd, "quit")) {
            break;
        }
        ZeroMemory(buffer, sizeof(buffer));
        Tips();
    }
}

int Read(ifstream& infile, char* buffer) {
    // 从文件中读取需要发送的数据
    if (infile.eof()) {
        return 0;
    }
    infile.read(buffer, 100);
    int cnt = infile.gcount();
    buffer[cnt] = 0;
    return cnt;
}

// 发送ack并且向右侧移动。
int Deliver(char* file, int ack) {
    while (recvWindow[ack].used) {
        recvWindow[ack].used = false; // 当前在接受窗口的最左侧，并且已经确认接受了
                                      // 窗口向右移动并且重置为false
        strcat_s(file, strlen(file) + strlen(recvWindow[ack].buffer) + 1, recvWindow[ack].buffer);
        ack++;
        ack %= SEQ_SIZE;
    }
    return ack;
}

int Send(ifstream& infile, int seq, SOCKET socket, SOCKADDR* addr) {
    // 发送数据
    clock_t now = clock();
    // 发送窗口大小为10   ACK的序号列表为20
    for (int i = 0; i < SWIN_SIZE; i++) {
        int j = (seq + i) % SEQ_SIZE;
        if (sendWindow[j].start == -1L) { // 该包已经确认
            continue;
        }
        if (sendWindow[j].start == 0L) { // 开始计时，这里代表包是首次发送
                                         // 从buffer + 2 的位置开始拷贝
            if (Read(infile, sendWindow[j].buffer + 2)) {
                sendWindow[j].start = now;
                sendWindow[j].buffer[0] = 20;
                sendWindow[j].buffer[1] = j + 1;
            } else if (i == 0) {
                return 0;
            } else {
                break;
            }
        } else if (now - sendWindow[j].start >= 1000L) { // 更新时间
                                                         // 一秒钟没有收到ACK，重发，这里重新设置发送时间
            sendWindow[j].start = now;
        } else {
            continue;
        }
        printf("Send: seq = %2d, data = %s\n", j + 1, sendWindow[j].buffer + 2);
        sendto(socket, sendWindow[j].buffer, strlen(sendWindow[j].buffer) + 1, 0, addr, sizeof(SOCKADDR));
    }
    return 1;
}

int MoveSendWindow(int seq) {
    // 移动窗口
    while (sendWindow[seq].start == -1L) { // 这个包已经确认
        sendWindow[seq].start = 0L;        // 这里重置了，重置为未确认，方便重新使用。
        seq++;
        seq %= SEQ_SIZE;
    }
    return seq;
}