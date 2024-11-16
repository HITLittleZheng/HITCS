#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

#define SERVER_PORT 12340     // 服务器端口号
#define SERVER_IP "127.0.0.1" // 服务器IP 地址

#define SEQ_SIZE 20      // 序列号个数
#define SWIN_SIZE 10     // 发送窗口大小
#define RWIN_SIZE 10     // 接收窗口大小
#define BUFFER_SIZE 1024 // 缓冲区大小
#define LOSS_RATE 0.2    // 丢包率

SOCKET socketServer;    // 服务器socket
SOCKADDR_IN addrServer; // 服务器网络地址

using namespace std;
struct recv
{
    bool used;
    char buffer[BUFFER_SIZE];
    recv()
    {
        used = false;
        ZeroMemory(buffer, sizeof(buffer));
    }
} recvWindow[SEQ_SIZE];

struct send
{
    clock_t start; // 每个数据包有一个计时器
    char buffer[BUFFER_SIZE];
    send()
    {
        start = 0;
        ZeroMemory(buffer, sizeof(buffer));
    }
} sendWindow[SEQ_SIZE];

char cmdBuffer[50];       // 命令缓冲区
char buffer[BUFFER_SIZE]; // 大小为1024的缓冲区
char cmd[10];             // 命令
char fileName[40];        // 文件名
char filePath[50];        // 文件路径
char file[1024 * 1024];   // 文件内容指针
int len = sizeof(SOCKADDR);
int recvSize;

int Deliver(char *file, int ack);
int Send(ifstream &infile, int seq, SOCKET socket, SOCKADDR *addr);
int MoveSendWindow(int seq);
int Read(ifstream &infile, char *buffer);

// 主函数
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
    // AF_INET：IP7
    // SOCK_DGRAM：UDP协议
    SOCKET socketServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    // 设置为非阻塞模式
    int iMode = 1;
    // 设置服务器套接字的IO模式
    ioctlsocket(socketServer, FIONBIO, (u_long FAR *)&iMode);
    // 服务器网络地址
    SOCKADDR_IN addrServer;
    inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
    addrServer.sin_family = AF_INET;
    addrServer.sin_port = htons(SERVER_PORT);
    // 绑定端口@@@@@@@@客户端不需要绑定端口
    if (err = bind(socketServer, (SOCKADDR *)&addrServer, sizeof(SOCKADDR)))
    {
        err = GetLastError();
        printf("绑定端口 %d 失败，错误码: % d\n", SERVER_PORT, err);
        WSACleanup();
        return -1;
    }
    else
    {
        printf("绑定端口 %d 成功", SERVER_PORT);
    }

    // 客户端地址
    SOCKADDR_IN addrClient;
    int status = 0; // 自动机状态——初始化为0
    clock_t start;  // 计时器
    clock_t now;
    int seq;
    int ack;
    ofstream outfile;
    ifstream infile;
    // 进入接收状态，注意服务器主要处理的任务是接收客户机请求，共有上载和下载两种任务
    while (true)
    {
        // 服务器套接字从客户端地址接收信息，传入buffer
        // 收到的buffer的结构：'10'+cmd[10]+filename[40]
        recvSize = recvfrom(socketServer, buffer, BUFFER_SIZE, 0, ((SOCKADDR *)&addrClient), &len);
        if ((float)rand() / RAND_MAX < LOSS_RATE)
        { // 设置丢包
            recvSize = 0;
            buffer[0] = 0; // 清空buffer首位的状态字节
        }
        switch (status) // 进入自动机状态的循环
        {
        case 0: // 接收请求
            if (recvSize > 0 && buffer[0] == 10)
            {
                char addr[100];
                ZeroMemory(addr, sizeof(addr)); // 全置零
                inet_ntop(AF_INET, &addrClient.sin_addr, addr, sizeof(addr));

                // buffer中将命令与文件名用数组分隔开
                sscanf_s(buffer + 1, "%s%s", cmd, sizeof(cmd) - 1, fileName, sizeof(fileName) - 1);

                if (strcmp(cmd, "upload") && strcmp(cmd, "download"))
                    continue; // 判断命令是否合法
                strcpy_s(filePath, "./");
                strcat_s(filePath, fileName); // 获取文件路径

                printf("收到来自客户端 %s 的请求: %s\n", addr, buffer); // 去除请求判断
                printf("是否同意该请求(Y/N)?");
                gets_s(cmdBuffer, 50); // 命令缓冲区
                if (!strcmp(cmdBuffer, "Y"))
                {
                    buffer[0] = 100;               // 缓冲区首字节的状态字
                    strcpy_s(buffer + 1, 3, "OK"); // 将指令写入缓冲区
                    if (!strcmp(cmd, "upload"))
                    {
                        file[0] = 0; // 如果是要上传
                        start = clock();
                        ack = 0;
                        status = 1;
                        outfile.open(filePath); // 打开/创建对应文件
                    }
                    else if (!strcmp(cmd, "download"))
                    {
                        start = clock();
                        seq = 0;
                        status = -1;
                        infile.open(filePath); // 如果要下载就打开对应文件
                    }
                }
                else
                {
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "NO");
                }
                // 把首字节为100的buffer发回去，开始通信
                sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
            }
            break;
        case 1: // 客户机请求上传，也就是服务器端是接收方
            if (recvSize > 0)
            {
                if (buffer[0] == 10)
                { // 状态字节
                    if (!strcmp(buffer + 1, "Finish"))
                    {
                        printf("传输完毕...\n");
                        start = clock();
                        sendWindow[0].start = start - 1000L;
                        sendWindow[0].buffer[0] = 100;
                        strcpy_s(sendWindow[0].buffer + 1, 3, "OK");
                        outfile.write(file, strlen(file));
                        status = 2;
                    }
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "OK");
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                }
                else if (buffer[0] == 20)
                { // 接收到一个数据包
                    seq = buffer[1];
                    int temp = seq - 1 - ack; // 步长
                    if (temp < 0)
                    {
                        temp += SEQ_SIZE;
                    }
                    start = clock();
                    seq--;
                    if (temp < RWIN_SIZE)
                    { // 步长小于窗口大小，可以继续发送
                        if (!recvWindow[seq].used)
                        { // 如果接受窗口当前待发送可用
                            recvWindow[seq].used = true;
                            strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
                        }
                        if (ack == seq)
                        {                             // 如果当前接收到的数据帧是最小未接受序号
                            ack = Deliver(file, ack); // 这一块数据帧可以写入文件，并且向前滑动窗口
                        }
                    }
                    printf("接收数据帧 seq = %d, data = %s, 发送 ack = %d, 起始 ack = %d\n", seq + 1, buffer + 2, seq + 1, ack + 1);
                    buffer[0] = 101; // 返回ack信息
                    buffer[1] = seq + 1;
                    buffer[2] = 0;
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                }
            }
            break;
        case 2: // 接收完成
            if (recvSize > 0 && buffer[0] == 10 && !strcmp(buffer + 1, "OK"))
            { // 状态码为0，且不是开始通信
                printf("传输成功，结束通信\n");
                status = 0;
                outfile.close();
            }
            now = clock();
            if (now - sendWindow[0].start >= 1000L)
            {
                sendWindow[0].start = now;
                sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
            }
            break;
        case -1: // 客户机请求下载，也就是服务器端充当发送方
            if (recvSize > 0)
            {
                if (buffer[0] == 10)
                { // 返回开始通信的状态码
                    if (!strcmp(buffer + 1, "OK"))
                    {
                        printf("开始传输...\n");
                        start = clock();
                        status = -2;
                    }
                    buffer[0] = 100;
                    strcpy_s(buffer + 1, 3, "OK");
                    sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
                }
            }
            // break;
            continue;
        case -2: // 服务器端发送数据
            if (recvSize > 0 && buffer[0] == 11)
            { // 接受到ack
                start = clock();
                ack = buffer[1];
                ack--;
                sendWindow[ack].start = -1L; // 收到了ack则重置，不超时
                if (ack == seq)
                {
                    seq = MoveSendWindow(seq);
                }
                printf("接收 ack = %d, 当前起始 seq = %d\n", ack + 1, seq + 1);
            }
            if (!Send(infile, seq, socketServer, (SOCKADDR *)&addrClient))
            { // 到文件结尾
                printf("传输完毕...\n");
                status = -3;
                start = clock();
                sendWindow[0].buffer[0] = 100;
                strcpy_s(sendWindow[0].buffer + 1, 7, "Finish"); // 传输结束信息
                sendWindow[0].start = start - 1000L;
            }
            break;
        case -3: // 请求完成
            if (recvSize > 0 && buffer[0] == 10)
            {
                if (!strcmp(buffer + 1, "OK"))
                {
                    printf("传输成功，结束通信\n");
                    infile.close();
                    status = 0;
                    break;
                }
            }
            now = clock();
            if (now - sendWindow[0].start >= 1000L)
            {
                sendWindow[0].start = now;
                sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR *)&addrClient, sizeof(SOCKADDR));
            }
        default:
            break;
        }
        if (status != 0 && clock() - start > 20000L)
        {
            printf("通信超时, 结束通信\n");
            status = 0;
            outfile.close();
            continue;
        }
        if (recvSize <= 0)
        { // 没有接收到信息时需要休眠
            Sleep(20);
        }
    }
    // 关闭套接字，卸载库
    closesocket(socketServer);
    WSACleanup();
    return 0;
}
int Read(ifstream &infile, char *buffer)
{
    // 从文件中读取需要发送的数据
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
    // 发送数据
    clock_t now = clock();
    for (int i = 0; i < SWIN_SIZE; i++)
    {
        int j = (seq + i) % SEQ_SIZE;
        if (sendWindow[j].start == -1L)
        { // 还未发送
            continue;
        }
        if (sendWindow[j].start == 0L)
        { // 超时，重新加载buffer
            if (Read(infile, sendWindow[j].buffer + 2))
            { // 从文件中读取该数据帧存储的内容
                sendWindow[j].start = now;
                sendWindow[j].buffer[0] = 200;   // 表示发送数据帧
                sendWindow[j].buffer[1] = j + 1; // 序号
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
        { // 更新时间
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
{ // 当发送序号与ack序号相等时移动，因此只需要判断是否超时而不需要判断发送未ack的情况
    // 移动窗口
    while (sendWindow[seq].start == -1L)
    {
        sendWindow[seq].start = 0L;
        seq++;
        seq %= SEQ_SIZE;
    }
    return seq;
}
