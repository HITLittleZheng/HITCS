#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

#define SERVER_PORT 12340     // 接收数据的端口号
#define SERVER_IP "127.0.0.1" // 服务器的 IP 地址
#define BUFFER_SIZE 1024      // 缓冲区大小
#define SEQ_SIZE 20           // 序列号个数
#define SWIN_SIZE 10          // 发送窗口大小
#define RWIN_SIZE 10          // 接收窗口大小
#define LOSS_RATE 0.2         // 丢包率

SOCKET socketServer;    // 服务器socket
SOCKADDR_IN addrServer; // 服务器网络地址
SOCKET socketClient;
using namespace std;

char cmdBuffer[50];
char buffer[BUFFER_SIZE];
char cmd[10];
char fileName[40];
char filePath[50];
char file[1024 * 1024];
int len = sizeof(SOCKADDR);
int recvSize;
int Deliver(char *file, int ack);
int Send(ifstream &infile, int seq, SOCKET socket, SOCKADDR *addr);
int MoveSendWindow(int seq);
int Read(ifstream &infile, char *buffer);
struct Cache
{
    bool used;
    char buffer[BUFFER_SIZE];
    Cache()
    {
        used = false;
        ZeroMemory(buffer, sizeof(buffer));
    }
} recvWindow[SEQ_SIZE];
struct DataFrame
{
    clock_t start;
    char buffer[BUFFER_SIZE];
    DataFrame()
    {
        start = 0;
        ZeroMemory(buffer, sizeof(buffer));
    }
} sendWindow[SEQ_SIZE];

int main(int argc, char *argv[])
{

    // 加载套接字库（必须）
    WORD wVersionRequested;
    WSADATA wsaData;
    // 套接字加载时错误提示
    int err;
    // 版本 2.2
    wVersionRequested = MAKEWORD(2, 2);
    // 加载 dll 文件 Socket 库
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0)
    {
        // 找不到 winsock.dll
        printf("加载 winsock 失败， 错误代码为: %d\n", WSAGetLastError());
        return FALSE;
    }
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
    {
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup();
        return FALSE;
    }
    else
        printf("The Winsock 2.2 dll was found okay\n");

    // 创建服务器套接字
    socketClient = socket(AF_INET, SOCK_DGRAM, 0);

    // 设置为非阻塞模式
    int iMode = 1;
    // 设置客户端服务器套接字的IO模式
    ioctlsocket(socketClient, FIONBIO, (u_long FAR *)&iMode);
    SOCKADDR_IN addrServer;
    inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
    addrServer.sin_family = AF_INET;
    addrServer.sin_port = htons(SERVER_PORT);

    int status = 0;
    clock_t start;
    clock_t now;
    int seq;
    int ack;
    srand((unsigned)time(NULL));
    while (true)
    {
        gets_s(cmdBuffer, 50); // 先从键盘输入缓冲区，再从缓冲区中读取
        sscanf_s(cmdBuffer, "%s%s", cmd, sizeof(cmd) - 1, fileName, sizeof(fileName) - 1);

        if (!strcmp(cmd, "upload"))
        {
            printf("申请上传文件: %s\n", fileName);
            strcpy_s(filePath, "./");
            strcat_s(filePath, fileName);
            ifstream infile(filePath);
            start = clock(); // 开始计时
            seq = 0;
            status = 0; // 初始化自动机的状态为0

            sendWindow[0].buffer[0] = 10;
            strcpy_s(sendWindow[0].buffer + 1, strlen(cmdBuffer) + 1, cmdBuffer);
            sendWindow[0].start = start - 1000L;
            // 进入通信循环
            while (true)
            {
                // 等待从服务器接收的信息
                // 从服务器接收信息写入客户端的buffer
                recvSize = recvfrom(socketClient, buffer, BUFFER_SIZE, 0, (SOCKADDR *)&addrServer, &len);
                switch (status)
                {
                case 0:
                    if (recvSize > 0 && buffer[0] == 100)
                    { // 状态字节，表示控制
                        if (!strcmp(buffer + 1, "OK"))
                        {
                            printf("通信开始，开始上传文件\n");
                            start = clock();
                            status = 1;
                            sendWindow[0].start = 0L; // 重新开始计时
                            continue;
                        }
                        else if (!strcmp(buffer + 1, "NO"))
                        {
                            status = -1;
                            break;
                        }
                    }
                    now = clock();
                    if (now - sendWindow[0].start >= 1000L)
                    { // 如果超时
                        sendWindow[0].start = now;
                        //
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 1: // 开始上传
                    if (recvSize > 0 && buffer[0] == 101)
                    { // 表示ack
                        start = clock();
                        ack = buffer[1];             // 接收缓冲区的位置
                        ack--;                       // 错位
                        sendWindow[ack].start = -1L; // 重置
                        if (ack == seq)
                        {
                            seq = MoveSendWindow(seq); // 移动窗口
                        }
                        printf("收到 ack = %d, 当前 seq = %d\n", ack + 1, seq + 1);
                    }
                    if (!Send(infile, seq, socketClient, (SOCKADDR *)&addrServer))
                    {
                        printf("文件传输结束\n");
                        status = 2; // 进入状态2
                        start = clock();
                        sendWindow[0].buffer[0] = 10;
                        strcpy_s(sendWindow[0].buffer + 1, 7, "Finish");
                        sendWindow[0].start = start - 1000L;
                        continue;
                    }
                    break;
                case 2:
                    if (recvSize > 0 && buffer[0] == 100)
                    {
                        if (!strcmp(buffer + 1, "OK"))
                        {
                            buffer[0] = 10;
                            strcpy_s(buffer + 1, 3, "OK");
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                            status = 3;
                            break;
                        }
                    }
                    now = clock();
                    if (now - sendWindow[0].start >= 1000L)
                    {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                    }
                default:
                    break;
                }
                if (status == -1)
                {
                    printf("服务器拒绝请求\n");
                    infile.close();
                    break;
                }
                if (status == 3)
                {
                    printf("上传成功，结束通信\n");
                    infile.close();
                    break;
                }
                if (clock() - start >= 10000L)
                {
                    printf("通信超时，结束通信\n");
                    infile.close();
                    break;
                }
                if (recvSize <= 0)
                {
                    Sleep(200);
                }
            }
        }
        else if (!strcmp(cmd, "download"))
        {
            printf("申请下载文件 %s\n", fileName);
            strcpy_s(filePath, "./");
            strcat_s(filePath, fileName);
            ofstream outfile(filePath);
            start = clock();
            ack = 0;
            status = 0;
            sendWindow[0].buffer[0] = 10; // 状态字节为0
            strcpy_s(sendWindow[0].buffer + 1, strlen(cmdBuffer) + 1, cmdBuffer);
            sendWindow[0].start = start - 1000L; // 重置
            while (true)
            {
                // 从服务器读到
                recvSize = recvfrom(socketClient, buffer, BUFFER_SIZE, 0, (SOCKADDR *)&addrServer, &len);
                if ((float)rand() / RAND_MAX < LOSS_RATE)
                { // 设置丢包
                    recvSize = 0;
                    buffer[0] = 0;
                }
                switch (status)
                {
                case 0:
                    if (recvSize > 0 && buffer[0] == 100)
                    {
                        if (!strcmp(buffer + 1, "OK"))
                        {
                            printf("申请通过, 准备下载...\n");
                            start = clock();
                            status = 1;
                            sendWindow[0].buffer[0] = 10;
                            strcpy_s(sendWindow[0].buffer + 1, 3, "OK");
                            sendWindow[0].start = start - 1000L;
                            continue;
                        }
                        else if (!strcmp(buffer + 1, "NO"))
                        {
                            status = -1;
                            break;
                        }
                    }
                    now = clock();
                    if (now - sendWindow[0].start >= 1000L)
                    {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 1:
                    if (recvSize > 0 && (unsigned char)buffer[0] == 200)
                    {
                        printf("开始下载...\n");
                        start = clock();
                        seq = buffer[1];
                        printf("接收数据帧 seq = %d, data = %s, 发送ack = %d\n", seq, buffer + 2, seq);
                        seq--;
                        recvWindow[seq].used = true;
                        strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                        if (ack == seq)
                        {
                            ack = Deliver(file, ack);
                        }
                        status = 2;
                        buffer[0] = 11;
                        buffer[1] = seq + 1;
                        buffer[2] = 0;
                        sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                        continue;
                    }
                    now = clock();
                    if (now - sendWindow[0].start >= 1000L)
                    {
                        sendWindow[0].start = now;
                        sendto(socketClient, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                    }
                    break;
                case 2:
                    if (recvSize > 0)
                    {
                        if ((unsigned char)buffer[0] == 200)
                        {
                            seq = buffer[1];
                            int temp = seq - 1 - ack;
                            if (temp < 0)
                            {
                                temp += SEQ_SIZE;
                            }
                            start = clock();
                            seq--;
                            if (temp < RWIN_SIZE)
                            {
                                if (!recvWindow[seq].used)
                                {
                                    recvWindow[seq].used = true;
                                    strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                                }
                                if (ack == seq)
                                {
                                    ack = Deliver(file, ack);
                                }
                            }
                            printf("接收数据帧 seq = %d, data = %s, 发送 ack = %d\n", seq + 1, buffer + 2, seq + 1, ack + 1);
                            buffer[0] = 11;
                            buffer[1] = seq + 1;
                            buffer[2] = 0;
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                        }
                        else if (buffer[0] == 100 && !strcmp(buffer + 1, "Finish"))
                        {
                            status = 3;
                            outfile.write(file, strlen(file));
                            buffer[0] = 10;
                            strcpy_s(buffer + 1, 3, "OK");
                            sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrServer, sizeof(SOCKADDR));
                            continue;
                        }
                    }
                    break;
                default:
                    break;
                }
                if (status == -1)
                {
                    printf("服务器拒绝请求\n");
                    outfile.close();
                    break;
                }
                if (status == 3)
                {
                    printf("下载成功, 结束通信\n");
                    outfile.close();
                    break;
                }
                if (clock() - start >= 20000L)
                {
                    printf("通信超时, 结束通信\n");
                    outfile.close();
                    break;
                }
                if (recvSize <= 0)
                {
                    Sleep(20);
                }
            }
        }
        else if (!strcmp(cmd, "quit"))
        {
            break;
        }
    }
    closesocket(socketClient);
    printf("关闭套接字\n");
    WSACleanup();
    return 0;
}

int Read(ifstream &infile, char *buffer)
{
    if (infile.eof())
    {
        return 0;
    }
    infile.read(buffer, 1000);
    int cnt = infile.gcount();
    buffer[cnt] = 0;
    return cnt;
}

int Deliver(char *file, int ack)
{
    while (recvWindow[ack].used)
    {
        recvWindow[ack].used = false;
        strcat_s(file, strlen(file) + strlen(recvWindow[ack].buffer) + 1, recvWindow[ack].buffer);
        ack++;
        ack %= SEQ_SIZE;
    }
    return ack;
}

int Send(ifstream &infile, int seq, SOCKET socket, SOCKADDR *addr)
{
    clock_t now = clock();
    for (int i = 0; i < SWIN_SIZE; i++)
    {
        int j = (seq + i) % SEQ_SIZE;
        if (sendWindow[j].start == -1L)
        {
            continue;
        }
        if (sendWindow[j].start == 0L)
        {
            if (Read(infile, sendWindow[j].buffer + 2))
            {
                sendWindow[j].start = now;
                sendWindow[j].buffer[0] = 20;
                sendWindow[j].buffer[1] = j + 1;
            }
            else if (i == 0)
            {
                return 0;
            }
            else
            {
                break;
            }
        }
        else if (now - sendWindow[j].start >= 1000L)
        {
            sendWindow[j].start = now;
        }
        else
        {
            continue;
        }
        printf("发送数据帧 seq = %d, data = %s\n", j + 1, sendWindow[j].buffer + 2);
        sendto(socket, sendWindow[j].buffer, strlen(sendWindow[j].buffer) + 1, 0, addr, sizeof(SOCKADDR));
    }
    return 1;
}

int MoveSendWindow(int seq)
{
    while (sendWindow[seq].start == -1L)
    {
        sendWindow[seq].start = 0L;
        seq++;
        seq %= SEQ_SIZE;
    }
    return seq;
}
