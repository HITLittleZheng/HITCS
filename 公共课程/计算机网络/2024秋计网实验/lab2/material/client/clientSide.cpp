#include <stdlib.h>
#include <WinSock2.h>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#pragma comment(lib, "ws2_32.lib")
#pragma warning(disable : 4996)

#define SERVER_PORT 12340       // 接收数据的端口号
#define SERVER_IP   "127.0.0.1" // 服务器的 IP 地址

using namespace std;

const int BUFFER_LENGTH = 1027; // 缓冲区大小
const int SEQ_SIZE = 20;        // 接收端序列号个数，为 1~20
BOOL ack[SEQ_SIZE];             // 存储收到的 ACK 状态，true 表示已确认
int curSeq;                     // 当前数据包的序列号
int curAck;                     // 当前等待确认的 ACK
int totalSeq;                   // 已接收的数据包总数
int totalPacket;                // 需要发送的数据包总数
int waitSeq;                    // 等待接收的序列号
const int SEND_WIND_SIZE = 10;  // 发送窗口大小
/****************************************************************/
/* -time 从服务器端获取当前时间
-quit 退出客户端
-testgbn [X] 测试 GBN 协议实现可靠数据传输
[X] [0,1] 模拟数据包丢失的概率 packetLossRatio,
[Y] [0,1] 模拟 ACK 丢失的概率 ackLossRatio,
*/
/****************************************************************/
void printTips() {
    printf("===========================================================================\n");
    printf("| -time to get current time |\n");
    printf("| -quit to exit client |\n");
    printf("| gbn + [packetLossRatio](0-1) +[ackLossRatio](0-1) + op  +filename |\n");
    printf("| sr + [packetLossRatio](0-1) +[ackLossRatio](0-1) + op  +filename  |\n");
    printf("===========================================================================\n");
}

//************************************
// Method: seqIsAvailable
// Access: public
// Returns: BOOL
// Qualifier: 判断序列号是否在窗口中，返回TRUE则说明在其中
// Parameter: nothing
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
// Access: public
// Returns: void
// Qualifier: 超时处理（seq回退重传）
// Parameter: nothing
//************************************
void timeoutHandler() {
    printf("Timer out error.\n");
    int index;
    for (int i = 0; i < (curSeq - curAck + SEQ_SIZE) % SEQ_SIZE; ++i) {
        index = (i + curAck) % SEQ_SIZE;
        ack[index] = TRUE;
    }
    totalSeq -= ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE);
    curSeq = curAck;
}
//************************************
// Method: ackHandler
// Access: public
// Returns: void
// Qualifier: 接收到ack并处理
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
// FullName: lossInLossRatio
// Access: public
// Returns: BOOL
// Qualifier: 根据丢失率随机生成一个数字，判断是否丢失,丢失则返回TRUE，否则返回 FALSE
// Parameter: float lossRatio [0,1]
//************************************
BOOL lossInLossRatio(float lossRatio) {
    int lossBound = (int)(lossRatio * 100);
    int r = rand() % 100;
    if (r < lossBound) {
        return TRUE;
    }
    return FALSE;
}

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

// 设置clientSocket为非阻塞模式
void setNonBlocking(SOCKET clientSocket) {
    int iMode = 1;
    ioctlsocket(clientSocket, FIONBIO, (u_long FAR *)&iMode);
}

// 设置clientSocket为阻塞模式
void setBlocking(SOCKET clientSocket) {
    int iMode = 0;
    ioctlsocket(clientSocket, FIONBIO, (u_long FAR *)&iMode);
}

void handleGBNDownload(SOCKET clientSocket, SOCKADDR_IN serverAddr, char *buffer, char *filename, float packetLossRatio, float ackLossRatio, int length) {
    sendto(clientSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&serverAddr, sizeof(SOCKADDR));
    setBlocking(clientSocket);
    char data[1024 * 113];
    int loct = 0;
    int stage = 0;
    bool transferIsDone = false;
    unsigned short recvSeq; // 接收窗口大小为 1，已确认的序列号
    while (true) {
        // 参数0的作用 TODO:
        recvfrom(clientSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&serverAddr, &length);
        switch (stage) {
        case 0:
            if ((unsigned char)buffer[0] == 205) {
                printf("准备进行文件传输\n");
                buffer[0] = 200;
                buffer[1] = '\0';
                sendto(clientSocket, buffer, 2, 0, (SOCKADDR *)&serverAddr, sizeof(SOCKADDR));
                stage = 1;
                loct = 0;
                recvSeq = 0; // 标志接受的窗口大小为1，已确认的序列号
                waitSeq = 1; // 等待接受的序列号
            }
            break;

        case 1:
            unsigned short seq = (unsigned short)buffer[0]; // 1-20
            BOOL b = lossInLossRatio(packetLossRatio);
            if (b) {
                printf("丢失序列号为 %d 的数据包\n", seq - 1);
                continue;
            }
            printf("接收到序列号为 %d 的数据包\n", seq - 1);
            // 如果是期待的包，正确接收，正常确认即可
            if (waitSeq == seq) {
                // buffer 偏移为2 loct为当前的包数，每个包含数据1024字节
                memcpy(data + 1024 * loct, buffer + 2, 1024);
                if (buffer[1] == '0') {
                    transferIsDone = true;
                }
                ++loct;
                ++waitSeq; // 这个序号按照1-20来计算的，因为这是期望
                if (waitSeq == 21) {
                    waitSeq = 1;
                }
                buffer[0] = seq;
                recvSeq = seq;
                buffer[2] = '\0';
            } else {
                if (!recvSeq) {
                    continue;
                }
                buffer[0] = recvSeq;
                buffer[1] = (unsigned char)buffer[1];
                buffer[2] = '\0';
            }
            b = lossInLossRatio(ackLossRatio);
            if (b) {
                printf("模拟丢失序列号为 %d 的ACK\n", (unsigned char)buffer[0] - 1);
                continue;
            }
            sendto(clientSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&serverAddr, sizeof(SOCKADDR));
            printf("发送ACK序列号为 %d\n", (unsigned char)buffer[0] - 1); //
        }
        if (transferIsDone) {
            printf("传输完成\n");
            setNonBlocking(clientSocket);
            break;
        }
    }
    ofstream ofs;
    ofs.open(filename, std::ios::out | std::ios::binary); // 以二进制模式写入文件
    char buff[1024];                                      // 修改为 1024 字节的缓冲区，匹配读取时的大小
    // 遍历所有读取的数据块
    for (int i = 0; i < loct; ++i) {
        // 将当前数据块复制到缓冲区
        memset(buff, 0, sizeof(buff)); // 清空缓冲区，防止残留数据影响
        memcpy(buff, data + 1024 * i, 1024);
        // 计算实际有效的数据长度，避免多余内容被写入
        std::size_t dataSize = strnlen(buff, 1024); // 获取有效数据长度
        // 写入有效数据到文件
        ofs.write(buff, dataSize);
    }
    ofs.close();
}

void processRequestToServer(SOCKET clientSocket, SOCKADDR_IN serverAddr) {

    char buffer[BUFFER_LENGTH];
    ZeroMemory(buffer, sizeof(buffer));
    int length = sizeof(SOCKADDR);
    char cmd[10];
    char operation[10];
    char filename[100];
    float packetLossRatio = 0.2; // 默认包丢失率
    float ackLossRatio = 0.2;    // 默认ACK丢失率
    // gbn 0.2 0.2 download testdownload.txt
    int ret; // sscanf 返回值
    srand((unsigned)time(NULL));
    while (true) {
        int flag = 0;
        printTips();
        fgets(buffer, BUFFER_LENGTH, stdin);
        int result = sscanf(buffer, "%s %f %f %s %s", &cmd, &packetLossRatio, &ackLossRatio, &operation, &filename);

        if (strcmp(cmd, "gbn") == 0) {
            flag = 1;
            printf("开始GBN协议的测试，请不要暂停进程\n");
            printf("丢包率为%.2f,ack丢包率为% .2f\n", packetLossRatio, ackLossRatio);
            if (strcmp(operation, "download") == 0) {
                handleGBNDownload(clientSocket, serverAddr, buffer, filename, packetLossRatio, ackLossRatio, length);
            } else if (strcmp(operation, "upload") == 0) {
                // handleGBNUpload(clientSocket, serverAddr, filename, packetLossRatio, ackLossRatio, length);
            }
        } else if (strcmp(cmd, "sr") == 0) {
            flag = 1;
            if (strcmp(operation, "download") == 0) {
                // handleSRDownload(clientSocket, serverAddr, filename, packetLossRatio, ackLossRatio, length);
            } else if (strcmp(operation, "upload") == 0) {
                // handleSRUpload(clientSocket, serverAddr, filename, packetLossRatio, ackLossRatio, length);
            }
        }
        // -time -quit 之类的直接发过去就好了
        if (flag)
            continue;
        sendto(clientSocket, buffer, strlen(buffer) + 1, 0, (SOCKADDR *)&serverAddr, sizeof(SOCKADDR));
        ret = recvfrom(clientSocket, buffer, BUFFER_LENGTH, 0, (SOCKADDR *)&serverAddr, &length);
        printf("%s\n", buffer);
        // 检查 buffer 的内容
        if (strcmp(buffer, "Good bye!") == 0) {
            break;
        }
    }
}

int main() {
    // initialize network
    if (initializeNetwork() == -1) {
        return -1;
    }

    // 创建的是一个数据报（Datagram）套接字。 指定为UDP协议
    SOCKET clientSocket = socket(AF_INET, SOCK_DGRAM, 0); // 创建一个UDP套接字
    // 阻塞与非阻塞应当在上传或者下载的时候进行具体的设置
    // int iMode = 1; // 1：非阻塞，0：阻塞
    // ioctlsocket(clientSocket, FIONBIO, (u_long FAR *)&iMode);  // 非阻塞设置
    if (clientSocket == INVALID_SOCKET) { // 创建失败
        printf("Create socket failed with error: %d\n", WSAGetLastError());
        return -1;
    }

    sockaddr_in serverAddr;          // 服务器地址
    serverAddr.sin_family = AF_INET; // 设置地址族为IPv4
    // 允许来自任意ip的访问
    serverAddr.sin_addr.S_un.S_addr = inet_addr(SERVER_IP); // 设置服务器地址为本地所有IP
    serverAddr.sin_port = htons(SERVER_PORT);

    processRequestToServer(clientSocket, serverAddr);

    closesocket(clientSocket);
    WSACleanup();
    return 0;
}