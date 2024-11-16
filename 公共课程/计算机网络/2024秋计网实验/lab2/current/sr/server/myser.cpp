#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")
#define SERVER_PORT 12345     // 端口号
#define SERVER_IP   "0.0.0.0" // IP 地址
#define SEQ_SIZE    20        // 序列号个数
#define SWIN_SIZE   10        // 发送窗口大小
#define RWIN_SIZE   10        // 收到窗口大小
#define BUFFER_SIZE 1024      // 缓冲区大小
#define LOSS_RATE   0.2       // 丢包率
using namespace std;
struct recv {
    bool used;
    char buffer[BUFFER_SIZE];
    recv() {
        used = false;
        ZeroMemory(buffer, sizeof(buffer));
    }
} recvWindow[SEQ_SIZE];

struct send {
    clock_t start; // 由于使用的是SR，因此每一个窗口位置都需要设置一个计时器
    char buffer[BUFFER_SIZE];
    send() {
        start = 0;
        ZeroMemory(buffer, sizeof(buffer));
    }
} sendWindow[SEQ_SIZE];

char cmdBuffer[50];
char buffer[BUFFER_SIZE];
char cmd[10];
char fileName[40];
char filePath[50];
char file[1024 * 1024];
int len = sizeof(SOCKADDR);
int recvSize;
int Deliver(char* file, int ack);
int Send(ifstream& infile, int seq, SOCKET socket, SOCKADDR* addr);
int MoveSendWindow(int seq);
int Read(ifstream& infile, char* buffer);
void getCurTime(char* ptime);

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

int initailizeNetwork() {
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

void InitializeVariables(int& seq, int& ack, int& stage, clock_t& start, clock_t& now, char* file) {
    seq = 0;
    ack = 0;
    stage = 0;
    now = 0;
    start = 0;
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

// 主函数
int main(int argc, char* argv[]) {
    int err;
    err = initailizeNetwork();
    // 创建服务器套接字
    SOCKET socketServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    // 设置为非阻塞模式
    setNonBlocking(socketServer);
    SOCKADDR_IN addrServer;
    inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
    addrServer.sin_family = AF_INET;
    addrServer.sin_port = htons(SERVER_PORT);

    // 一些初始化
    srand((unsigned)time(NULL));

    // 绑定端口
    // 使服务器能够在指定的端口上监听和接收客户端的请求
    if (err = bind(socketServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR))) {
        err = GetLastError();
        WSACleanup();
        return -1;
    } else {
        printf("成功监听端口Port： %2d \n", SERVER_PORT);
    }

    SOCKADDR_IN addrClient;
    int stage = 0;
    clock_t start;
    clock_t now;
    int seq;
    int ack;
    ofstream outfile;
    ifstream infile;
    while (true) {
        recvSize = recvfrom(socketServer, buffer, BUFFER_SIZE, 0, ((SOCKADDR*)&addrClient), &len);

        switch (stage) {
        case 0: // 收到请求
            if (recvSize > 0 && buffer[0] == 10) {
                char addr[100];
                ZeroMemory(addr, sizeof(addr));
                inet_ntop(AF_INET, &addrClient.sin_addr, addr, sizeof(addr));
                sscanf_s(buffer + 1, "%s%s", cmd, sizeof(cmd) - 1, fileName, sizeof(fileName) - 1);
                if (recvSize > 0 && !strcmp(cmd, "time")) {
                    char ptime[100];
                    getCurTime(ptime);
                    printf("Current time: %s\n", ptime);
                    buffer[0] = 10;
                    strcpy_s(buffer + 1, strlen(ptime) + 1, ptime);
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
                    ZeroMemory(buffer, sizeof(buffer));
                    continue;
                }
                if (strcmp(cmd, "up") && strcmp(cmd, "dl")) {
                    continue;
                }
                strcpy_s(filePath, "./");
                strcat_s(filePath, fileName);
                if (1) {
                    // attention
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "OK");
                    if (!strcmp(cmd, "up")) {
                        InitializeVariables(seq, ack, stage, start, now, file);
                        file[0] = 0;
                        start = clock();
                        ack = 0;
                        stage = 1;
                        outfile.open(filePath);
                    } else if (!strcmp(cmd, "dl")) {
                        InitializeVariables(seq, ack, stage, start, now, file);
                        start = clock();
                        seq = 0;
                        stage = -1;
                        infile.open(filePath); // 这里确认了要下载的文件
                    }
                } else {
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "NO");
                }
                // 接收到客户端的首次请求
                // 返回buffer[0] = 100, buffer[1] = "OK"
                sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
            }
            break;
        case 1: // 客户机请求上传，也就是服务器端是收到方
            if ((recvSize > 0) && ((float)rand() / RAND_MAX < LOSS_RATE)) {
                printf("出现丢包\n");
                recvSize = 0;
                buffer[0] = 0;
            }
            if (recvSize > 0) {
                if (buffer[0] == 10) {
                    if (!strcmp(buffer + 1, "Finish")) {
                        printf("传输完成\n");
                        start = clock();
                        sendWindow[0].start = start - 1000L;
                        sendWindow[0].buffer[0] = 100;
                        strcpy_s(sendWindow[0].buffer + 1, 3, "OK");
                        outfile.write(file, strlen(file));
                        stage = 2;
                    }
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "OK");
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
                } else if (buffer[0] == 20) {
                    seq = buffer[1];
                    int temp = seq - 1 - ack;
                    if (temp < 0) {
                        temp += SEQ_SIZE;
                    }
                    start = clock();
                    seq--;
                    if (temp < RWIN_SIZE) {
                        if (!recvWindow[seq].used) {
                            recvWindow[seq].used = true;
                            strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                        }
                        if (ack == seq) {
                            ack = Deliver(file, ack);
                        }
                    }
                    printf("Recv: seq = %2d, data = %s\n", seq + 1, buffer + 2);
                    printf("Send: ack = %2d, Start ack = %2d\n", seq + 1, ack + 1);
                    buffer[0] = 101;
                    buffer[1] = seq + 1;
                    buffer[2] = 0;
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
                }
            }
            break;
        case 2: // 收到完成
            if (recvSize > 0 && buffer[0] == 10 && !strcmp(buffer + 1, "OK")) {
                printf("传输完成\n");
                stage = 0;
                outfile.close();
            }
            now = clock();
            if (now - sendWindow[0].start >= 1000L) {
                sendWindow[0].start = now;
                sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
            }
            break;
        case -1: // 客户机请求下载，也就是服务器端充当发送方
            if (recvSize > 0) {
                if (buffer[0] == 10) {
                    // 第二次进入Attention逻辑
                    if (!strcmp(buffer + 1, "OK")) {
                        printf("开始文件传输\n");
                        start = clock();
                        stage = -2;
                    }
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "OK");
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
                }
            }
            break;
        case -2: // 服务器端发送数据
            if (recvSize > 0 && buffer[0] == 11) {
                start = clock();
                ack = buffer[1];
                ack--;                       // 接收到的是序号。需要-1
                sendWindow[ack].start = -1L; // 接收到ack，重置此窗口
                if (ack == seq) {
                    seq = MoveSendWindow(seq); // 这里重置了 start为0L
                }
                printf("Recv: ack = %2d, Curr seq = %2d\n", ack + 1, seq + 1);
            }
            // 这里的Send会发送200状态码
            // 完整的文件发送过程。
            // 上面的if并非第一次进入case-2的时候就执行，其实是先在这里执行了Send逻辑
            if (!Send(infile, seq, socketServer, (SOCKADDR*)&addrClient)) {
                printf("传输结束\n");
                stage = -3;
                start = clock();
                sendWindow[0].buffer[0] = 100;
                strcpy_s(sendWindow[0].buffer + 1, 7, "Finish");
                sendWindow[0].start = start - 1000L;
            }
            break;
        case -3: // 请求完成
            if (recvSize > 0 && buffer[0] == 10) {
                if (!strcmp(buffer + 1, "OK")) {
                    // printf("传输结束\n");
                    infile.close();
                    stage = 0;
                    break;
                }
            }
            now = clock();
            if (now - sendWindow[0].start >= 1000L) {
                sendWindow[0].start = now;
                sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
            }
        default:
            break;
        }
        if (stage != 0 && clock() - start > 5000L) {
            printf("Timeout\n");
            stage = 0;
            infile.close();
            outfile.close();
            continue;
        }
        if (recvSize <= 0) {
            Sleep(20);
        }
        ZeroMemory(buffer, sizeof(buffer));
    }
    // 关闭套接字，卸载库
    closesocket(socketServer);
    WSACleanup();
    return 0;
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
int Deliver(char* file, int ack) { // 处理ack。判断当前ack是否处理，就将数据拼接到file中
    while (recvWindow[ack].used) {
        recvWindow[ack].used = false; // 处理过就重置
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
                sendWindow[j].buffer[0] = 200;
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

void getCurTime(char* ptime) {
    char buffer[128];
    memset(buffer, 0, sizeof(buffer));
    time_t c_time;
    struct tm* p;
    time(&c_time);
    p = localtime(&c_time);
    sprintf(buffer, "%d/%d/%d %d:%d:%d", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
    strcpy(ptime, buffer);
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
