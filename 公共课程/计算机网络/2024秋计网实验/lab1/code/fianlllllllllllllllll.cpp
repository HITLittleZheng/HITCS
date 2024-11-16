#include <Winsock2.h> // 包含 Winsock2 头文件
#include <Ws2tcpip.h> // 包含 inet_ntop 函数的头文件
#include <cstdio>     // 标准输入输出库
#include <direct.h>   // 用于Windows下创建文件夹
#include <fstream>    // 用于文件读写
#include <functional> // 用于哈希函数
#include <iostream>   // 包含标准输入输出流
#include <process.h>  // 包含线程相关的函数
#include <sstream>    // 字符串流操作
#include <string.h>   // 包含字符串处理的函数
#include <sys/stat.h> // 添加文件夹相关函数
#include <tchar.h>    // 包含 Unicode 和 ANSI 兼容函数
#include <cstring>
#include <cstdlib> // 包含malloc和free
using namespace std;

// 编译命令：g++ http_server.c -o http_server -lws2_32
#pragma comment(lib, "ws2_32.lib") // 链接ws2_32.lib库，编译时需要添加 -lws2_32

#define MAXSIZE 114514 // 定义最大报文长度
#define HTTP_PORT 80   // 定义HTTP协议的端口号

#define FISH_WEBSITE_FROM "cs.hit.edu.cn"   // 钓鱼网站源网址
#define FISH_WEBSITE_TO "jwes.hit.edu.cn"   // 钓鱼网站目的网址
#define FISH_WEBSITE_HOST "jwes.hit.edu.cn" // 钓鱼目的地址的主机名

const BOOL FISH_WEBSITE_INTRO = TRUE; // 是否开启钓鱼网站引导

// 全局定义代理服务器套接字、地址信息和端口号
SOCKET ProxyServer;
sockaddr_in ProxyServerAddr;
const int ProxyPort = 10240;
// const int IPnum = 10; // IP 地址数量（全局）

// 缓存相关参数
boolean needCache = TRUE; // 判断是否需要缓存
// 定义HTTP请求头
struct HttpHeader
{
    char method[8];         // 定义HTTP请求方式
    char url[1024];         // 定义URL
    char host[1024];        // 定义主机名 这里应该是那个url
    char cookie[1024 * 10]; // 定义Cookie
    HttpHeader()
    {                                         // 构造函数
        ZeroMemory(this, sizeof(HttpHeader)); // 将结构体的内存空间清零
    }
};

// 定义一个列表，判断是否禁止访问某个网站
const int HostNum = 0; // 被禁止访问的IP地址数量
const char *ForbiddenHost[HostNum] = {
    // "http.p2hp.com",
    // "jwes.hit.edu.cn",
};

const int ForbiddenIPCount = 5;
const char *ForbiddenIPs[ForbiddenIPCount] = {
    "192.168.1.100", // 示例禁止的IP
    "192.168.1.101", "10.0.0.1", "203.0.113.5", "198.51.100.2",
    
    // "127.0.0.1"
};

// 本机服务器套接字  复用
struct ProxyParam
{
    // 客户端套接字
    SOCKET clientSocket;
    // 目标服务器套接字
    SOCKET serverSocket;
};

DWORD __stdcall ProxyThread(LPVOID lpParameter);
BOOL InitSocket();
void ParseHttpHead(char *buffer, HttpHeader *httpHeader);
bool IsIPForbidden(const char *clientIP);

// 在初始化阶段创建缓存文件夹
void CreateCacheDirectory()
{
    if (_mkdir("cache") == 0)
    {
        printf("创建缓存目录成功\n");
    }
    else if (GetLastError() != ERROR_ALREADY_EXISTS)
    {
        printf("创建缓存目录失败\n");
    }
}

// 使用 std::hash 生成 URL 哈希
std::string HashUrl(const std::string &url)
{
    std::hash<std::string> hasher;
    size_t hashValue = hasher(url);
    std::stringstream ss;
    ss << std::hex << hashValue; // 转换为十六进制字符串
    return ss.str();
}

// 入口函数
int _tmain(int argc, _TCHAR *argv[])
{
    // 设置控制台输出和输入代码页为 UTF-8
    CreateCacheDirectory();
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    printf("代理服务器正在启动\n");
    printf("初始化...\n");
    if (!InitSocket())
    {
        printf("socket 初始化失败\n");
        return -1;
    }

    printf("套接字初始化成功\n "
           "监听端口：%d\n===============================================\n\n",
           ProxyPort);
    SOCKET clientSocket = INVALID_SOCKET;
    ProxyParam *lpProxyParam;
    HANDLE hThread; // 句柄：和对象一一对应的32位无符号整数值

    // 监听
    while (true)
    {
        SOCKADDR_IN clientAddr;
        int addrLen = sizeof(clientAddr);
        // 接受一个新的连接
        clientSocket = accept(ProxyServer, (SOCKADDR *)&clientAddr, &addrLen);
        if (clientSocket == INVALID_SOCKET)
        {
            printf("接受连接失败，错误代码：%d\n", WSAGetLastError());
            continue;
        }
        char clientIP[INET_ADDRSTRLEN];
        // 获取客户端IP地址
        inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIP, INET_ADDRSTRLEN);
        printf("客户端IP地址: %s\n", clientIP);

        // 检查客户端IP是否在禁止列表中
        if (IsIPForbidden(clientIP))
        {
            printf("禁止IP地址访问: %s\n", clientIP);
            closesocket(clientSocket);
            continue; // 拒绝连接
        }

        lpProxyParam = new ProxyParam;
        lpProxyParam->clientSocket = clientSocket;
        // 创建一个新的线程来处理该连接
        hThread = CreateThread(NULL, 0, ProxyThread, lpProxyParam, 0, NULL);
        if (hThread == NULL)
        {
            printf("创建线程失败，错误代码：%d\n", GetLastError());
            closesocket(clientSocket);
            delete lpProxyParam;
            continue;
        }
        // 设置线程超时时间为10秒
        // DWORD threadTimeout = 5000; // 10秒
        // DWORD waitResult = WaitForSingleObject(hThread, threadTimeout);

        // if (waitResult == WAIT_TIMEOUT)
        // {
        //     // printf("线程超时，强制结束线程。\n");
        //     TerminateThread(hThread, 0);
        // }
        // else if (waitResult == WAIT_OBJECT_0)
        // {
        //     // printf("线程在超时前结束。\n");
        // }
        // else
        // {
        //     // printf("等待线程时发生错误，错误代码：%d\n", GetLastError());
        // }

        CloseHandle(hThread); // 关闭线程句柄，不关闭线程，只是不再干扰这个线程
    }
}

bool IsIPForbidden(const char *clientIP)
{
    for (int i = 0; i < ForbiddenIPCount; i++)
    {
        if (strcmp(clientIP, ForbiddenIPs[i]) == 0)
        {
            return true; // IP在禁止列表中
        }
    }
    return false; // IP不在禁止列表中
}

//************************************
// Method: InitSocket
// FullName: InitSocket
// Access: public
// Returns: BOOL
// Qualifier: 初始化套接字
//************************************
BOOL InitSocket()
{
    // 定义变量用于加载套接字库
    // wsa: Windows Sockets API
    WORD wVersionRequested; // 请求的版本
    WSADATA wsaData;        // 存储关于套接字库的信息
    /*
    typedef struct WSAData {
        WORD wVersion;       // Winsock 版本
        WORD wHighVersion;   // 支持的最高 Winsock 版本
        char szDescription[WSADESCRIPTION_LEN + 1]; // 描述字符串
        char szSystemStatus[WSASYS_STATUS_LEN + 1]; // 系统状态字符串
        unsigned short iMaxSockets; // 最大套接字数
        unsigned short iMaxUdpDg;   // 最大UDP数据报大小
        char* lpVendorInfo;         // 供应商信息
    } WSADATA, *LPWSADATA;
    */
    int err; // 用于存储错误代码
    // 设置请求的 Winsock 版本为 2.2
    wVersionRequested = MAKEWORD(2, 2);

    // 调用 WSAStartup 函数加载 Winsock DLL
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0)
    {
        // 如果加载失败，打印错误代码
        printf("加载 winsock 失败，错误代码为: %d\n", WSAGetLastError());
        return FALSE; // 返回失败状态
    }

    // 检查加载的版本是否为 2.2
    // 主版本号2高字节，次版本号2低字节
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
    {
        // 如果版本不匹配，打印错误信息并清理
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup(); // 清理已加载的 Winsock
        return FALSE; // 返回失败状态
    }

    /* socket() 函数：用于创建套接字
        - int af：地址族，当前支持 AF_INET（IPv4）或 AF_INET6（IPv6）
        - int type：套接字类型，SOCK_STREAM 表示顺序、可靠、双向的连接流
        - int protocol：协议，值为 0 时由服务提供商选择合适的协议
    */
    ProxyServer = socket(AF_INET, SOCK_STREAM, 0);

    // 检查套接字创建是否成功
    if (INVALID_SOCKET == ProxyServer)
    {
        // 如果创建失败，打印错误代码
        printf("创建套接字失败，错误代码为： %d\n", WSAGetLastError());
        return FALSE; // 返回失败状态
    }

    // 设置套接字地址结构
    ProxyServerAddr.sin_family = AF_INET;        // 设置地址族为 IPv4
    ProxyServerAddr.sin_port = htons(ProxyPort); // 将端口号转换为网络字节序
    // 绑定到本机的所有 IP 地址（可选择绑定到特定 IP）
    // ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;  // 允许来自所有 IP
    // 的连接 ProxyServerAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1"); //
    // 仅允许本机访问
    ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;
    // 绑定套接字与指定的网络地址
    // ProxyServerAddr取引用得到的是结构体的地址，强制转换成 SOCKADDR* 类型
    if (bind(ProxyServer, (SOCKADDR *)&ProxyServerAddr, sizeof(SOCKADDR)) ==
        SOCKET_ERROR)
    {
        // 如果绑定失败，打印错误信息
        printf("绑定套接字失败\n");
        return FALSE; // 返回失败状态
    }

    // 开始监听套接字，等待客户端连接
    if (listen(ProxyServer, SOMAXCONN) == SOCKET_ERROR)
    {
        // 如果监听失败，打印错误信息
        printf("监听端口%d 失败", ProxyPort);
        return FALSE; // 返回失败状态
    }

    // 返回成功状态
    return TRUE;
}

//************************************
// Method: ParseHttpHead
// FullName: ParseHttpHead
// Access: public
// Returns: void
// Qualifier: 解析 TCP 报文中的 HTTP 头部
// Parameter: char * buffer
// Parameter: HttpHeader * httpHeader
//************************************
void ParseHttpHead(char *buffer, HttpHeader *httpHeader)
{
    char *p;                    // 指向分割后的字符串
    char *ptr;                  // 存储分割函数的上下文信息
    const char *delim = "\r\n"; // 定义换行符
    p = strtok_s(
        buffer, delim,
        &ptr); // 调用 strtok_s 函数从 buffer 指向的字符串中分割出第一个子字符串
    // printf("%s\n", p);
    if (p == NULL)
    {
        return;
    }
    // 检测 CONNECT 方法 检测https请求
    if (p[0] == 'C' && strstr(p, "CONNECT") == p)
    {
        // printf("检测到 CONNECT 方法\n");
        memcpy(httpHeader->method, "CONNECT", 7);
        // printf("httpHeader->method: %s111\n", httpHeader->method);
        // 解析 CONNECT 请求中的主机和端口
        char *urlStart = strstr(p, " ") + 1;
        char *urlEnd = strstr(urlStart, " ");
        memcpy(httpHeader->url, urlStart, urlEnd - urlStart);
        return;
    }
    if (p[0] == 'G')
    { // GET 方式
        memcpy(httpHeader->method, "GET", 3);
        // 从p[4]的位置开始往后取 url长度的字符串
        memcpy(httpHeader->url, &p[4], strlen(p) - 13);
    }
    else if (p[0] == 'P')
    { // POST 方式
        memcpy(httpHeader->method, "POST", 4);
        // 从p[5]的位置开始往后取 url长度的字符串
        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
    }
    // 获取http请求头中的第二行: 需要做出判断
    // 本实验只接受了 Host | cookie
    p = strtok_s(NULL, delim, &ptr); // 调用 strtok_s 函数从 buffer
                                     // 指向的字符串中继续分割出下一个子字符串
    // p不为null的时候，继续循环
    while (p)
    {
        switch (p[0])
        {
        case 'H': // 表示 HTTP 请求头中的主机名字段
            // 6 表示 "Host: "的长度 包括了空格
            memcpy(httpHeader->host, &p[6], strlen(p) - 6);
            break;
        case 'C': // 表示 HTTP 请求头中可能有 Cookie 字段
            if (strlen(p) > 8)
            {
                char header[8];
                ZeroMemory(header, sizeof(header));
                memcpy(header, p, 6);
                if (!strcmp(header, "Cookie"))
                {
                    memcpy(httpHeader->cookie, &p[8], strlen(p) - 8);
                }
            }
            break;
        default:
            break;
        }
        // 继续读取下一行
        p = strtok_s(NULL, delim, &ptr);
    }
}

// 判断是否禁止访问某个主机
bool IsForbidden(const HttpHeader &httpHeader)
{
    for (int i = 0; i < HostNum; i++)
    {
        if (strcmp(httpHeader.host, ForbiddenHost[i]) == 0)
        {
            return true; // 如果匹配，则禁止访问
        }
    }
    return false; // 否则允许访问
}

// 接受来自客户端的数据，
bool ReceiveData1(SOCKET socket, char **buffer, int &recvLen)
{
    *buffer = (char *)malloc(MAXSIZE);
    ZeroMemory(*buffer, MAXSIZE);                // 清空缓冲区
    recvLen = recv(socket, *buffer, MAXSIZE, 0); // 接收数据
    if (recvLen == SOCKET_ERROR)
    {
        printf("ReceiveData1接收数据失败，错误代码：%d\n", WSAGetLastError());
        return false; // 返回失败状态
    }
    // 打印接受的数据
    // printf("接收到数据：\n%s\n", buffer);
    return true; // 返回成功状态
}

// 接受来自服务器段的数据
bool ReceiveData2(SOCKET socket, char *&buffer, int &totalLen)
{
    const int TEMP_SIZE = 8096; // 临时缓冲区大小，用于接收头部数据
    char tempBuffer[TEMP_SIZE];
    ZeroMemory(tempBuffer, TEMP_SIZE);

    // 初步读取数据头部
    int recvLen = recv(socket, tempBuffer, TEMP_SIZE - 1, 0);
    if (recvLen <= 0)
    {
        printf("ReceiveData2接收数据失败，错误代码：%d\n", WSAGetLastError());
        return false;
    }

    tempBuffer[recvLen] = '\0'; // 确保以NULL结尾

    // 查找 "Content-Length" 字段并解析
    int contentLength = 0;
    const char *contentLengthStr = "Content-Length: ";
    char *pos = strstr(tempBuffer, contentLengthStr);
    bool isChunked = strstr(tempBuffer, "Transfer-Encoding: chunked") != nullptr;

    if (pos)
    {
        contentLength = atoi(pos + strlen(contentLengthStr));
    }

    // 如果是chunked传输，或者没有Content-Length，则设置为接收未知长度
    int headerSize = strstr(tempBuffer, "\r\n\r\n") - tempBuffer + 4;

    // buffer = nullptr;

    // if (isChunked)
    // {
    //     // 处理 chunked 传输
    //     buffer = (char *)malloc(TEMP_SIZE); // 初始分配缓冲区
    //     totalLen = 0;
    //     // 复制已读取的头部
    //     memcpy(buffer, tempBuffer, recvLen);
    //     totalLen += recvLen;
    //     // 继续处理每个chunk
    //     bool flag = recvLen >= TEMP_SIZE - 1;
    //     while (true && flag)
    //     {
    //         char chunkSizeBuffer[16];
    //         ZeroMemory(chunkSizeBuffer, sizeof(chunkSizeBuffer));
    //         int chunkSizeLen = recv(socket, chunkSizeBuffer, sizeof(chunkSizeBuffer) - 1, 0);
    //         if (chunkSizeLen <= 0)
    //         {
    //             break;
    //         }
    //         chunkSizeBuffer[chunkSizeLen] = '\0';
    //         // 提取 chunk size，有效字符应为十六进制数字
    //         char *validChunkSizeStr = chunkSizeBuffer;
    //         while (*validChunkSizeStr == ' ' || *validChunkSizeStr == '\r' || *validChunkSizeStr == '\n')
    //         {
    //             validChunkSizeStr++; // 跳过无关字符
    //         }
    //         // 解析chunk大小
    //         int chunkSize = 0;
    //         char *endPtr = nullptr;
    //         // Ensure that the chunkSizeBuffer only contains valid hexadecimal characters by trimming any trailing characters like \r\n
    //         // for (int i = 0; i < sizeof(chunkSizeBuffer); i++)
    //         // {
    //         //     if (chunkSizeBuffer[i] == '\r' || chunkSizeBuffer[i] == '\n')
    //         //     {
    //         //         chunkSizeBuffer[i] = '\0'; // Replace with null terminator to clean the buffer
    //         //         break;
    //         //     }
    //         // }
    //         chunkSize = strtol(validChunkSizeStr, &endPtr, 16);

    //         // Check if the conversion was successful and the chunkSize is valid
    //         if (endPtr == chunkSizeBuffer || chunkSize < 0)
    //         {
    //             printf("解析chunk大小失败，chunkSizeBuffer: %s\n", chunkSizeBuffer);
    //             break;
    //         }
    //         if (chunkSize == 0)
    //         {
    //             break; // End of chunks
    //         }
    //         // 读取chunk数据
    //         buffer = (char *)realloc(buffer, totalLen + chunkSize + 2);
    //         int bytesReceived = recv(socket, buffer + totalLen, chunkSize + 2, 0); // 包括\r\n
    //         if (bytesReceived <= 0)
    //         {
    //             break;
    //         }
    //         totalLen += bytesReceived;
    //     }
    //     printf("接收了chunked传输的总长度为: %d 字节\n", totalLen);
    //     return true;
    // }
    if (contentLength > 0)
    {
        // 如果有 Content-Length
        totalLen = headerSize + contentLength;
        buffer = (char *)malloc(totalLen + 1);
        if (!buffer)
        {
            printf("内存分配失败\n");
            return false;
        }

        ZeroMemory(buffer, totalLen + 1);
        memcpy(buffer, tempBuffer, recvLen); // 复制已经读取的头部

        // 接收剩余的数据
        int remaining = contentLength - (recvLen - headerSize);
        int offset = recvLen;
        while (remaining > 0)
        {
            recvLen = recv(socket, buffer + offset, remaining, 0);
            if (recvLen <= 0)
            {
                printf("接收剩余数据失败，错误代码：%d\n", WSAGetLastError());
                free(buffer);
                return false;
            }
            offset += recvLen;
            remaining -= recvLen;
        }

        printf("接收到的数据长度为: %d 字节\n", totalLen);
        return true;
    }
    else
    {
        // 如果没有Content-Length和chunked传输，持续接收直到连接关闭
        buffer = (char *)malloc(TEMP_SIZE); // 初始分配缓冲区
        totalLen = 0;

        // 复制已读取的头部
        memcpy(buffer, tempBuffer, recvLen);
        totalLen += recvLen;

        printf("接收了未指定长度的总长度为: %d 字节\n", totalLen);
        return true;
    }
}

// 发送数据到目标服务器
bool SendData(SOCKET socket, const char *buffer, int len)
{
    if (send(socket, buffer, len, 0) == SOCKET_ERROR)
    {
        printf("发送数据失败，错误代码：%d\n", WSAGetLastError());
        return false; // 返回失败状态
    }
    return true; // 返回成功状态
}

// 创建并连接目标服务器
SOCKET CreateAndConnectSocket(const char *host)
{
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0); // 创建套接字
    if (serverSocket == INVALID_SOCKET)
    {
        printf("创建套接字失败，错误代码：%d\n", WSAGetLastError());
        return INVALID_SOCKET; // 返回无效套接字
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(HTTP_PORT); // 设置端口
    HOSTENT *hostent = gethostbyname(host); // 获取主机信息
    if (!hostent)
    {
        // printf("获取主机信息失败\n");
        closesocket(serverSocket); // 关闭套接字
        return INVALID_SOCKET;     // 返回无效套接字
    }
    // serverAddr.sin_addr.S_un.S_addr = inet_addr(host); // 设置服务器地址
    in_addr Inaddr = *((in_addr *)*hostent->h_addr_list);
    serverAddr.sin_addr.s_addr = inet_addr(inet_ntoa(Inaddr));
    printf("连接目标服务器：%s\n", host);
    if (connect(serverSocket, (SOCKADDR *)&serverAddr, sizeof(serverAddr)) ==
        SOCKET_ERROR)
    {
        printf("连接服务器失败，错误代码：%d\n", WSAGetLastError());
        closesocket(serverSocket); // 关闭套接字
        return INVALID_SOCKET;     // 返回无效套接字
    }
    printf("连接主机 %s 成功\n", host);
    return serverSocket; // 返回有效的服务器套接字
}

// 清理并关闭套接字
void Cleanup(SOCKET clientSocket, SOCKET serverSocket, ProxyParam *param)
{
    closesocket(clientSocket); // 关闭客户端套接字
    closesocket(serverSocket); // 关闭服务器套接字
    delete param;              // 释放参数
    _endthreadex(0);           // 结束线程
}

// 读取缓存文件的最后修改时间
std::string GetLastModifiedTime(const std::string &cacheFileName)
{
    struct stat result;
    if (stat(cacheFileName.c_str(), &result) == 0)
    {
        char timeBuffer[80];
        struct tm *timeinfo = gmtime(&result.st_mtime);
        strftime(timeBuffer, sizeof(timeBuffer), "%a, %d %b %Y %H:%M:%S GMT",
                 timeinfo);
        return std::string(timeBuffer);
    }
    return "";
}

// 构造带有 If-Modified-Since 头的 HTTP 请求
void AddIfModifiedSinceHeader(char *buffer, const std::string &lastModified)
{
    std::string modifiedHeader = "If-Modified-Since: " + lastModified + "\r\n";
    std::string request(buffer);
    size_t pos = request.find("\r\n\r\n"); // 找到请求头的结束位置
    if (pos != std::string::npos)
    {
        request.insert(pos + 2, modifiedHeader);
        strcpy(buffer, request.c_str()); // 将修改后的请求重新写入 buffer
    }
}

bool IsNotModified(char *responseBuffer)
{
    // 检查服务器的响应状态码
    return strstr(responseBuffer, "304 Not Modified") != NULL;
}

// 检查缓存文件是否存在
bool IsCacheExist(const std::string &cacheFileName)
{
    struct stat buffer;
    return (stat(cacheFileName.c_str(), &buffer) == 0);
}

// 读取缓存数据
bool ReadCache(const std::string &cacheFileName, char *buffer, int &len)
{
    std::ifstream cacheFile(cacheFileName, std::ios::binary);
    if (!cacheFile.is_open())
    {
        return false;
    }
    cacheFile.seekg(0, std::ios::end);
    len = cacheFile.tellg();
    cacheFile.seekg(0, std::ios::beg);
    cacheFile.read(buffer, len);
    cacheFile.close();
    return true;
}

// 替换buffer中所有oldStr为newStr的函数
void replaceAll(char *buffer, const char *oldStr, const char *newStr)
{
    // 临时缓冲区，用于存储替换后的结果
    char temp[MAXSIZE];
    char *pos, *lastPos = buffer;
    int oldLen = strlen(oldStr);
    int newLen = strlen(newStr);

    // 初始化临时缓冲区
    temp[0] = '\0';

    // 查找oldStr在buffer中的位置
    while ((pos = strstr(lastPos, oldStr)) != nullptr)
    {
        // 将 lastPos 到 pos 之间的内容复制到 temp 中
        strncat(temp, lastPos, pos - lastPos);
        // 追加 newStr 到 temp 中
        strcat(temp, newStr);
        // 移动指针，跳过已经替换过的部分
        lastPos = pos + oldLen;
    }

    // 将剩余部分追加到 temp 中
    strcat(temp, lastPos);

    // 将替换后的内容复制回原来的 buffer 中
    strncpy(buffer, temp, MAXSIZE - 1);
    buffer[MAXSIZE - 1] = '\0'; // 确保buffer是以'\0'结尾的
}

// 代理线程的执行函数
DWORD __stdcall ProxyThread(LPVOID lpParameter)
{
    ProxyParam *param = (ProxyParam *)lpParameter;
    SOCKET clientSocket = param->clientSocket;
    SOCKET serverSocket = param->serverSocket;
    char *buffer = nullptr; // 定义缓冲区
    int recvLen = 0;        // 接收数据长度

    // 接收来自客户端的数据
    if (!ReceiveData1(clientSocket, &buffer, recvLen))
    {
        Cleanup(clientSocket, INVALID_SOCKET, param);
        return 0;
    }
    char *cacheBuffer = new char[recvLen + 1];
    memcpy(cacheBuffer, buffer, recvLen);
    HttpHeader httpHeader;
    ParseHttpHead(cacheBuffer, &httpHeader); // 解析 HTTP 头部
    delete[] cacheBuffer;
    // 判断是否为 HTTPS 流量 (CONNECT 方法)
    if (strcmp(httpHeader.method, "CONNECT") == 0)
    {
        // printf("检测到 HTTPS 请求，目标：%s\n", httpHeader.url);
        // 发送 "200 Connection Established" 响应
        const char *established = "HTTP/1.1 200 Connection Established\r\n\r\n";
        send(clientSocket, established, strlen(established), 0);
        // 不再处理后续流量，直接结束线程
        Cleanup(clientSocket, INVALID_SOCKET, param);
        return 0;
    }
    printf("接收到数据：\n%s\n", buffer);

    // 创建缓存文件名
    std::string url(httpHeader.url);
    std::string cacheFileName = "cache/" + HashUrl(url);

    // printf("接收到客户端数据：\n%s\n", buffer);
    // delete cacheBuffer;
    // 判断主机是否被禁止访问
    // printf("host: %s", httpHeader.host);
    if (IsForbidden(httpHeader))
    {
        printf("禁止访问此host\n");
        Cleanup(clientSocket, INVALID_SOCKET, param);
        return 0;
    }
    // 钓鱼网站引导
    if (strstr(httpHeader.url, FISH_WEBSITE_FROM) != NULL &&
        FISH_WEBSITE_INTRO == TRUE)
    {
        printf("\n=====================================\n");
        printf("-------------已从源网址：%s 转到 目的网址 ：%s ----------------\n",
               FISH_WEBSITE_FROM, FISH_WEBSITE_TO);
        printf("\n=====================================\n");
        // replaceAll(buffer, httpHeader.host, FISH_WEBSITE_HOST);
        memcpy(httpHeader.host, FISH_WEBSITE_HOST, strlen(FISH_WEBSITE_HOST) + 1);
        memcpy(httpHeader.url, FISH_WEBSITE_TO, strlen(FISH_WEBSITE_TO));

        cacheFileName = "cache/" + HashUrl(url);
    }
    // 连接目标服务器
    // printf("\n=====================================\n");
    printf("当前连接的服务器host为: %s\n", httpHeader.host);
    // printf("\n=====================================\n");
    serverSocket = CreateAndConnectSocket(httpHeader.host);
    // 检查缓存文件是否存在

    if (serverSocket == INVALID_SOCKET)
    {

        Cleanup(clientSocket, INVALID_SOCKET, param);
        return 0;
    }

    // 发送数据到目标服务器
    printf("发送数据到目标服务器, 将要发送的数据为：\n%s "
           "======================================\n\n",
           buffer);
    if (!SendData(serverSocket, buffer, recvLen))
    {
        Cleanup(clientSocket, serverSocket, param);
        return 0;
    }
    printf("发送数据到目标服务器成功\n");

    // 循环监听并转发目标服务器返回的数据
    // recvLen = recv(serverSocket, buffer, MAXSIZE, 0);
    // char append = 'a';
    // printf("如你所见，这是一步打印调试接收数据\n");
    recvLen = 0;
    if (ReceiveData2(serverSocket, buffer, recvLen) && recvLen != 0)
    {
        bool ifCacheSaved = false;
        // printf("到这里了吗"); 没到
        if (IsCacheExist(cacheFileName))
        {
            httpHeader.url[sizeof(httpHeader.url) - 1] = '\0'; // Ensure null-termination
            printf("url:%s 缓存文件存在，缓存文件名是%s\n", httpHeader.url, cacheFileName.c_str());
            // 判断buffer中的状态码是否为304，如果为304，则从本地缓存中读取数据返回给客户端
            if (IsNotModified(buffer))
            {
                printf("状态码为304，从缓存中读取数据\n");
                char CacheBuffer[MAXSIZE];
                int cacheLen;
                if (ReadCache(cacheFileName, CacheBuffer, cacheLen))

                {
                    // printf("转发数据到客户端, "
                    //        "将要转发的数据为：\n%s====================================="
                    //        "===\n\n",
                    //        CacheBuffer);
                    send(clientSocket, CacheBuffer, cacheLen, 0);
                }
            }
            else
            {
                printf("状态码不为304，将数据返回给客户端\n");
                // printf("转发数据到客户端, "
                //        "将要转发的数据为：\n%s======================================="
                //        "=\n\n",
                //        buffer);
                // 状态码不为304，直接返回数据
                // 写入缓存
                std::ofstream cacheFile(cacheFileName, std::ios::binary);
                printf("写入缓存文件：%s\n", cacheFileName.c_str());
                if (cacheFile.is_open() && needCache == TRUE)
                {
                    cacheFile.write(buffer, recvLen);
                    cacheFile.close();
                }
                send(clientSocket, buffer, recvLen, 0);
                ifCacheSaved = true;
            }
        }
        else
        {
            // 缓存不存在
            printf("缓存文件不存在，将数据返回给客户端\n");
            // printf("转发数据到客户端, "
            //        "将要转发的数据为：\n%s======================================="
            //        "=\n\n",
            //        buffer);
            // 发送数据到客户端
            SendData(clientSocket, buffer, recvLen); // 将数据发送回客户端
        }
         if(ifCacheSaved == false) {
            std::ofstream cacheFile(cacheFileName, std::ios::binary);
                printf("写入缓存文件：%s\n", cacheFileName.c_str());
                if (cacheFile.is_open() && needCache == TRUE)
                {
                    cacheFile.write(buffer, recvLen);
                    cacheFile.close();
                }
         }
    }
    printf("转发数据到客户端, "
                   "将要转发的数据为：\n%s======================================="
                   "=\n\n",
                   buffer);
    // printf("如你所见、这也是一部打印");
    // 清理资源
   
    Cleanup(clientSocket, serverSocket, param);
    return 0;
}