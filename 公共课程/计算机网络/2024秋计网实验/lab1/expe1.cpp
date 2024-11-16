#define _CRT_SECURE_NO_WARNINGS 1
#pragma comment(lib, "ws2_32.lib") // 链接ws2_32.lib库，似乎链接不了？可以通过命令行参数解决

#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0A00 // 指定目标版本为win11
#include <stdio.h>
#include <process.h>
#include <string.h>
#include <tchar.h>
#include <fstream>
#include <string>
#include <ws2tcpip.h>
#include <winsock2.h>
#include <windows.h>
// #include <Windows.h>
#include <iostream>
#include <chrono>
#include <sstream>
using namespace std;

#define MAXSIZE 165507 // 发送数据报文的最大长度
#define HTTP_PORT 80   // http 服务器端口
#define INET_ADDRSTRLEN 22
#define INET6_ADDRSTRLEN 65
// #define BANNED_WEB "lib.hit.edu.cn"                 // 屏蔽网站
#define PHISHING_WEB_SRC "jwc.hit.edu.cn"           // 钓鱼原网址
#define PHISHING_WEB_DEST "http://jwts.hit.edu.cn/" // 钓鱼目的网址
#define BANNED_NUM 3
const char *banedIP[BANNED_NUM] = {"192.168.0.1", "192.168.0.2", "192.168.0.3"};             // 屏蔽用户ip列表
const char *banedWEB[BANNED_NUM] = {"lib.hit.edu.cn", "lib1.hit.edu.cn", "lib2.hit.edu.cn"}; // 屏蔽网站列表
// Http请求头格式
struct HttpHeader
{
    char method[4];         // POST 或 GET
    char url[1024];         // 请求的 url
    char host[1024];        // 目标主机
    char cookie[1024 * 10]; // cookie
    HttpHeader()
    {
        ZeroMemory(this, sizeof(HttpHeader));
    }
};

// 缓存的报文格式
struct HttpCache
{
    char url[1024];
    char host[1024];
    char last_modified[200];
    char status[4];
    char buffer[MAXSIZE];
    HttpCache()
    {
        ZeroMemory(this, sizeof(HttpCache)); // 初始化cache
    }
};
HttpCache Cache[1024];
int cached_number = 0; // 已经缓存的url数
int last_cache = 0;    // 上一次缓存的索引

// 代理服务器存储的客户端与目标服务器套接字
struct ProxyParam
{
    SOCKET clientSocket;
    SOCKET serverSocket;
};

BOOL InitWsa();
BOOL InitSocket();
int getIpByUrl(char *url, char *ipstr);
void ParseCachedModified(char *buffer, char *status, char *last_modified);
int ParseHttpHead(char *buffer, HttpHeader *httpHeader, int *store);
BOOL ConnectToServer(SOCKET *serverSocket, char *url);
unsigned int __stdcall ProxyThread(LPVOID lpParameter);

// 代理相关参数
SOCKET ProxyServer;          // 代理服务器套接字
sockaddr_in ProxyServerAddr; // 代理服务器地址
const int ProxyPort = 10240; // 代理服务器端口

// 由于新的连接都使用新线程进行处理，对线程的频繁的创建和销毁特别浪费资源
// 可以使用线程池技术提高服务器效率
// const int ProxyThreadMaxNum = 20;
// HANDLE ProxyThreadHandle[ProxyThreadMaxNum] = {0};
// DWORD ProxyThreadDW[ProxyThreadMaxNum] = {0};
bool isBanned(char *ip)
{
    for (int i = 0; i < BANNED_NUM; i++)
    {
        if (strcmp(ip, banedIP[i]) == 0)
        {
            printf("IP %s 已被屏蔽\n", ip);
            return true;
        }
    }
    return false;
}
//************************************
//  Method: InitWsa
//  FullName: InitSWsa
//  Access: public
//  Returns: BOOL
//  Qualifier: 加载并确认Wsa
//************************************
BOOL InitWsa()
{
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(2, 2);
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0)
    {
        printf("加载 winsock 失败， 错误代码为: %d\n", WSAGetLastError());
        return FALSE;
    }
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
    {
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup();
        return FALSE;
    }
    return TRUE;
}

//************************************
// Method: InitSocket
// FullName: InitSocket
// Access: public
// Returns: BOOL
// Qualifier: 初始化代理服务器套接字，开始监听.
//************************************
BOOL InitSocket()
{
    ProxyServer = socket(AF_INET, SOCK_STREAM, 0); // IPv4, TCP
    if (INVALID_SOCKET == ProxyServer)
    {
        printf("创建套接字失败，错误代码为：%d\n", WSAGetLastError());
        return FALSE;
    }
    ProxyServerAddr.sin_family = AF_INET; // IPv4
    ProxyServerAddr.sin_port = htons(ProxyPort);
    // ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;
    ProxyServerAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    // 绑定端点地址和套接字
    if (bind(ProxyServer, (SOCKADDR *)&ProxyServerAddr, sizeof(SOCKADDR)) == SOCKET_ERROR)
    {
        printf("绑定套接字失败\n");
        return FALSE;
    }
    if (listen(ProxyServer, SOMAXCONN) == SOCKET_ERROR)
    {
        printf("监听端口%d 失败", ProxyPort);
        return FALSE;
    }
    return TRUE;
}

//*************************
// Method: ParseCache
// FullName: ParseCache
// Access: public
// Returns: void
// Qualifier: 解析响应的 HTTP 头部,提取出状态码和 Last-Modified，用于在 cache命中的情况下判断是否需要更新缓存
// Parameter: char *buffer
// Parameter: char * status
// Parameter: HttpHeader *httpHeader
//*************************
void ParseCachedModified(char *buffer, char *status, char *last_modified)
{
    char *p;
    char *ptr;
    int flag = 0;
    const char *delim = "\r\n";
    p = strtok_s(buffer, delim, &ptr);
    memcpy(status, &p[9], 3);
    status[3] = '\0';
    p = strtok_s(NULL, delim, &ptr);
    while (p)
    {
        if (strstr(p, "Last-Modified") != NULL)
        {
            flag = 1;
            memcpy(last_modified, &p[15], strlen(p) - 15);
            break;
        }
        p = strtok_s(NULL, delim, &ptr);
    }
    if (flag == 0)
        printf("服务器没有没有发送Last-Modified字段信息\n");
}

//*************************
// Method: ParseHttpHead
// FullName: ParseHttpHead
// Access: public
// Returns: int
// Qualifier: 解析 TCP 报文中的 HTTP 头部，判断是否命中缓存，在缓存中存储请求的 URL
// Parameter: char *buffer
// Parameter: HttpHeader *httpHeader
//*************************
int ParseHttpHead(char *buffer, HttpHeader *httpHeader, int *store)
{
    int flag = 0; // 用于判断是否命中 cache，1-命中，0-未命中
    char *p;
    char *ptr;
    const char *delim = "\r\n";
    // 处理请求行
    p = strtok_s(buffer, delim, &ptr);
    // 将 url 存入 cache
    if (p[0] == 'G') // GET方式
    {
        printf("GET请求\n");
        memcpy(httpHeader->method, "GET", 3);
        // 获取主路径
        int index = 0;
        while (p[index] != '/')
        {
            index++;
        }
        index += 2;
        int start = index;
        for (; index < strlen(p) && p[index] != '/'; index++)
        {
            httpHeader->url[index - start] = p[index];
        }
        httpHeader->url[index - start] = '\0';
        // printf("这个网站的摘出来的URLURLURLURLUUUUURRRRRLLLLLLL: %s\n", httpHeader->url);
        for (int i = 0; i < 1024; i++)
        {
            if (Cache[i].url[0] != '\0')
            {
                printf("Cache[%d].url: %s\n", i, Cache[i].url);
            }
            if (strcmp(Cache[i].url, httpHeader->url) == 0) // 已存在，更改flag退出循环
            {
                flag = 1;
                break;
            }
        }
        if (!flag && cached_number != 1023) // 不存在 && 当前轮cache未满，直接存储
        {
            memcpy(Cache[cached_number].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = cached_number; // 记录最后一次缓存的索引
        }
        else if (!flag && cached_number == 1023) // 不存在 && 当前轮cache已满，覆盖第一个
        {
            memcpy(Cache[0].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = 0;
        }
        store[0] = last_cache;
    }
    else if (p[0] == 'P') // POST方式
    {
        printf("POST请求\n");
        memcpy(httpHeader->method, "POST", 4);
        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
        for (int i = 0; i < 1024; i++)
        {
            if (strcmp(Cache[i].url, httpHeader->url) == 0)
            {
                flag = 1;
                break;
            }
        }
        if (!flag && cached_number != 1023)
        {
            memcpy(Cache[cached_number].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = cached_number;
        }
        else if (!flag && cached_number == 1023)
        {
            memcpy(Cache[0].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = 0;
        }
    }
    else if (p[0] == 'C') // CONNECT方式
    {
        printf("CONNECT请求\n");
        memcpy(httpHeader->method, "CONNECT", 7);
        memcpy(httpHeader->url, &p[8], strlen(p) - 21);
        // printf("url：%s\n", httpHeader->url);
        for (int i = 0; i < 1024; i++)
        {
            if (strcmp(Cache[i].url, httpHeader->url) == 0)
            {
                flag = 1;
                break;
            }
        }
        if (!flag && cached_number != 1023)
        {
            memcpy(Cache[cached_number].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = cached_number;
        }
        else if (!flag && cached_number == 1023)

        {
            memcpy(Cache[0].url, httpHeader->url, strlen(httpHeader->url));
            last_cache = 0;
        }
    }
    // 继续处理 Host 和 Cookie
    p = strtok_s(NULL, delim, &ptr);
    while (p)
    {
        switch (p[0])
        {
        case 'H': // HOST，存入 cache
            memcpy(httpHeader->host, &p[6], strlen(p) - 6);
            if (!flag && cached_number != 1023)
            {
                memcpy(Cache[last_cache].host, &p[6], strlen(p) - 6);
                cached_number++;
            }
            else if (!flag && cached_number == 1023)
            {
                memcpy(Cache[last_cache].host, &p[6], strlen(p) - 6);
            }
            break;
        case 'C': // Cookie
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
        p = strtok_s(NULL, delim, &ptr);
    }
    return flag;
}
//*************************
// Method: getIpByUrl
// FullName: getIpByUrl
// Access: public
// Returns: int
// Qualifier: 通过 url 获取 ip 地址
// Parameter: char *url
// Parameter: char *ipstr
//*************************
int getIpByUrl(char *url, char *ipstr)
{
    int iResult;
    // 初始化结构体
    struct addrinfo *result = NULL;
    struct addrinfo *ptr = NULL;
    struct addrinfo hints;

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // getaddrinfo
    iResult = getaddrinfo(url, NULL, &hints, &result);
    if (iResult != 0)
    {
        printf("getaddrinfo failed: %d\n", iResult);
        return 1;
    }
    // 解析地址
    for (ptr = result; ptr != NULL; ptr = ptr->ai_next)
    {
        if (ptr->ai_family == AF_INET)
        { // IPv4
            struct sockaddr_in *ipv4 = (struct sockaddr_in *)ptr->ai_addr;
            snprintf(ipstr, INET_ADDRSTRLEN, "%s", inet_ntoa(ipv4->sin_addr));
            printf("IPV4 Address: %s\n", ipstr);
        }
        else if (ptr->ai_family == AF_INET6)
        { // IPv6
            struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)ptr->ai_addr;
            inet_ntop(AF_INET6, &ipv6->sin6_addr, ipstr, INET6_ADDRSTRLEN);
            printf("IPV6 Address: %s\n", ipstr);
        }
        else
        {
            continue;
        }
    }
    freeaddrinfo(result);
    return 0;
}
//************************************
// Method: ConnectToServer
// FullName: ConnectToServer
// Access: public
// Returns: BOOL
// Qualifier: 连接目标服务器
// Parameter: SOCKET * serverSocket
// Parameter: char * url
//************************************
BOOL ConnectToServer(SOCKET *serverSocket, char *url)
{
    sockaddr_in serverAddr; // 服务器端口地址
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(HTTP_PORT);
    char ipstr[INET6_ADDRSTRLEN];
    int iResult = getIpByUrl(url, ipstr);
    if (iResult)
    {
        printf("解析目标服务器ip失败\n");
        return FALSE;
    }
    serverAddr.sin_addr.s_addr = inet_addr(ipstr);
    // 目标服务器的ip地址
    printf("目标服务器ip：%s\n", ipstr);
    *serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (*serverSocket == INVALID_SOCKET)
    {
        printf("创建面向目标服务器socket失败\n");
        return FALSE;
    }
    if (connect(*serverSocket, (SOCKADDR *)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        printf("连接目标服务器失败\n");
        closesocket(*serverSocket);
        return FALSE;
    }
    return TRUE;
}
//************************************
// Method: createAndWriteToFile
// FullName: createAndWriteToFile
// Access: public
// Returns: void
// Qualifier: 创建文件并写入字符串
// Parameter: const char * inputString
// Parameter: char * url
//************************************
void createAndWriteToFile(const char *inputString, char *url)
{
    char fileName[30] = "cache-";
    strncat(fileName, url, 23); // 用url名创建文件
    char filePath[256];
    snprintf(filePath, sizeof(filePath), ".\\cpp-code\\computer-net\\expe1\\buffers\\%s", fileName);
    FILE *file = fopen(filePath, "w");
    if (file != NULL)
    {
        fputs(inputString, file); // 将字符串写入文件
        fclose(file);             // 关闭文件
        printf("\n文件%s创建成功\n", fileName);
    }
    else
    {
        perror("文件打开失败文件打开失败文件打开失败文件打开失败");
    }
}
//************************************
// Method: ProxyThread
// FullName: ProxyThread
// Access: public
// Returns: unsigned int __stdcall
// Qualifier: 子线程执行函数。处理客户端请求，转发给目标服务器，接收目标服务器的响应，转发给客户端
// Parameter: LPVOID lpParameter
//************************************
unsigned int __stdcall ProxyThread(LPVOID lpParameter)
{
    printf("代理服务器线程启动\n");
    char Buffer[MAXSIZE];      // 收到字节流或发送字符串的缓存
    char phishBuffer[MAXSIZE]; // 钓鱼的构造报文缓存
    char *CacheBuffer;         // 字节流与字符串转换的缓存

    SOCKADDR_IN clientAddr; // 客户端端点信息
    int length = sizeof(SOCKADDR_IN);
    int recvSize;   // 代理服务器接收到的字节数
    int ret;        // 发送字节数
    int Have_cache; // 代理服务器是否有缓存
    HttpHeader *httpHeader = new HttpHeader();

    // 获取客户端的IP地址
    getpeername(((ProxyParam *)lpParameter)->clientSocket, (SOCKADDR *)&clientAddr, &length);
    char *clientIP = inet_ntoa(clientAddr.sin_addr);
    printf("客户端 IP: %s\n", clientIP);

    // 检查是否屏蔽该IP
    if (isBanned(clientIP))
    {
        printf("IP %s 被屏蔽，断开连接\n", clientIP);
        goto error; // 直接断开连接
    }
    // 清空残留数据
    ZeroMemory(Buffer, MAXSIZE);
    ZeroMemory(phishBuffer, MAXSIZE);
    recvSize = recv(((ProxyParam *)lpParameter)->clientSocket, Buffer, MAXSIZE, 0);

    if (recvSize <= 0)
    {
        printf("没有收到客户端报文\n");
        goto error;
    }
    printf("代理服务器从客户端接收到的报文是：\n");
    printf("%s", Buffer);
    // memcpy(sendBuffer, Buffer, recvSize);

    // 将字节流转换为字符串
    CacheBuffer = new char[recvSize + 1];
    ZeroMemory(CacheBuffer, recvSize + 1); // 确保最后是 \0
    memcpy(CacheBuffer, Buffer, recvSize);
    int store[1];
    // 解析HTTP头部，存入cache，step1
    Have_cache = ParseHttpHead(CacheBuffer, httpHeader, store);
    // printf("last_cache last_cache last_cache last_cache last_cache:%d\n", last_cache);
    delete CacheBuffer;

    // 为代理服务器创建套接字，连接目标服务器。
    if (!ConnectToServer(&((ProxyParam *)lpParameter)->serverSocket, httpHeader->url))
    {
        printf("代理服务器连接目标服务器 %s 失败\n", httpHeader->host);
        goto error;
    }
    printf("代理服务器连接目标服务器 %s 成功\n", httpHeader->host);

    for (int i = 0; i < BANNED_NUM; i++)
    {
        if (strcmp(httpHeader->host, banedWEB[i]) == 0)
        {
            // printf("网站 %s 已被屏蔽已被屏蔽已被屏蔽已被屏蔽已被屏蔽已被屏蔽\n", banedWEB[i]);
            printf("网站 %s 已被屏蔽\n", banedWEB[i]);
            goto error;
        }
    }

    // 网站钓鱼
    if (strstr(httpHeader->url, PHISHING_WEB_SRC) != NULL)
    {
        char *pr;
        printf("网站 %s 已被成功重定向至 %s\n", PHISHING_WEB_SRC, PHISHING_WEB_DEST);
        // 构造报文，设置 Location
        int phishing_len = snprintf(phishBuffer, sizeof(phishBuffer),
                                    "HTTP/1.1 302 Moved Temporarily\r\n"
                                    "Connection:keep-alive\r\n"
                                    "Cache-Control:max-age=0\r\n"
                                    "Location: %s\r\n\r\n",
                                    PHISHING_WEB_DEST);
        // 将拼接好的 302 报文发送给客户端
        ret = send(((ProxyParam *)lpParameter)->clientSocket, phishBuffer, phishing_len, 0);
        printf("成功发送钓鱼报文\n");
        goto error;
    }

    // 实现cache功能，step
    if (Have_cache) // 请求的页面在代理服务器有缓存
    {
        printf("非第一次访问该页面，存在缓存，正在验证是否需要替换缓存\n");
        // printf("请求的页面在代理服务器有缓存请求的页面在代理服务器有缓存请求的页面在代理服务器有缓存请求的页面在代理服务器有缓存\n");
        // printf("缓存的URL：%s\n", Cache[last_cache].url);
        char cached_buffer[MAXSIZE]; // 处理后发送的 HTTP 数据报文
        ZeroMemory(cached_buffer, MAXSIZE);
        memcpy(cached_buffer, Buffer, recvSize);

        char ifModifiedSinceHeader[MAXSIZE];
        // if (Cache[last_cache].last_modified[0] == '\0') // 没有上次修改时间
        // {
        //     char lastModified[] = "Thu, 01 Jan 1970 00:00:00 GMT";
        //     snprintf(ifModifiedSinceHeader, sizeof(ifModifiedSinceHeader), "If-Modified-Since: %s\r\n", lastModified);
        // }
        // else
        // {
        //     snprintf(ifModifiedSinceHeader, sizeof(ifModifiedSinceHeader), "If-Modified-Since: %s\r\n", Cache[last_cache].last_modified);
        // }
        snprintf(ifModifiedSinceHeader, sizeof(ifModifiedSinceHeader), "If-Modified-Since: %s\r\n", Cache[last_cache].last_modified);

        char *headerEnd = strstr(cached_buffer, "\r\n\r\n"); // 找到头部结束的位置
        if (headerEnd != nullptr)
        {
            int headerLength = headerEnd - cached_buffer + 2; // \r\n 的位置
            char newBuffer[MAXSIZE];
            memset(newBuffer, 0, MAXSIZE);
            strncpy(newBuffer, cached_buffer, headerLength);
            strncat(newBuffer, ifModifiedSinceHeader, strlen(ifModifiedSinceHeader));
            strcat(newBuffer, headerEnd + 2); // 从 \r\n\r\n 后的内容开始拷贝
            recvSize = strlen(newBuffer);
            memset(cached_buffer, 0, MAXSIZE);
            strncpy(cached_buffer, newBuffer, recvSize);
        }

        printf("发送的修改后的消息为\n%s\n", cached_buffer);
        // 发送处理后的报文给目标服务器
        ret = send(((ProxyParam *)lpParameter)->serverSocket, cached_buffer, strlen(cached_buffer) + 1, 0);
        recvSize = recv(((ProxyParam *)lpParameter)->serverSocket, cached_buffer, MAXSIZE, 0);
        if (recvSize <= 0)
        {
            printf("没有收到目标服务器对添加If-Modified-Since处理后的请求的回复\n");
            goto error;
        }
        char *headerPartPtr = strstr(cached_buffer, "\r\n\r\n");
        int length = headerPartPtr - cached_buffer + 4;
        char *headerPart = new char[length + 1];
        strncpy(headerPart, cached_buffer, length);
        headerPart[length] = '\0';
        printf("返回报文头为\n%s\n", headerPart);
        delete[] headerPart;
        CacheBuffer = new char[recvSize + 1];
        ZeroMemory(CacheBuffer, recvSize + 1);
        memcpy(CacheBuffer, cached_buffer, recvSize);
        char last_status[4];    // 记录主机返回的状态字
        char last_modified[30]; // 记录返回页面的修改时间
        ParseCachedModified(CacheBuffer, last_status, last_modified);
        printf("目标服务器上次修改时间%s\n", last_modified);
        delete CacheBuffer;

        if (strcmp(last_status, "304") == 0) // 304状态码，文件没有被修改
        {
            // printf("last_cache%s\n", Cache[last_cache].url);
            // printf("store%s\n", Cache[store[0]].url);
            printf("状态码304，页面未被修改\n");
            char url_for304[1024] = "304";
            strncat(url_for304, Cache[last_cache].url, 30);
            createAndWriteToFile(cached_buffer, url_for304);
            // 直接将缓存数据转发给客户端
            ret = send(((ProxyParam *)lpParameter)->clientSocket, Cache[last_cache].buffer, sizeof(Cache[last_cache].buffer), 0);

            if (ret != SOCKET_ERROR)
                printf("由缓存发送\n");
        }
        else if (strcmp(last_status, "200") == 0) // 200状态码，表示文件已被修改
        {
            // 修改缓存内容
            printf("last_cache%s\n", Cache[last_cache].url);
            printf("store%s\n", Cache[store[0]].url);
            printf("状态码200，页面被修改");
            memcpy(Cache[last_cache].buffer, cached_buffer, strlen(cached_buffer));        // 更新缓buffer
            memcpy(Cache[last_cache].last_modified, last_modified, strlen(last_modified)); // 更新修改时间
            createAndWriteToFile(cached_buffer, Cache[last_cache].url);
            // 发给客户端
            ret = send(((ProxyParam *)lpParameter)->clientSocket, Cache[last_cache].buffer, sizeof(Cache[last_cache].buffer), 0);
            if (ret != SOCKET_ERROR)
                printf("由缓存发送，已修改\n");
        }
    }
    else // 没有缓存过该页面
    {
        // 将客户端发送的 HTTP 数据报文直接转发给目标服务器
        // printf("没有缓存过该页面没有缓存过该页面没有缓存过该页面没有缓存过该页面没有缓存过该页面\n");
        printf("未命中，是第一次访问该页面，无缓存\n");
        ret = send(((ProxyParam *)lpParameter)->serverSocket, Buffer, strlen(Buffer) + 1, 0);
        recvSize = recv(((ProxyParam *)lpParameter)->serverSocket, Buffer, MAXSIZE, 0);
        if (recvSize <= 0)
        {
            printf("没有收到转发客户端请求后目标服务器的回复\n");
            goto error;
        }
        // 将目标服务器返回的数据直接转发给客户端
        ret = send(((ProxyParam *)lpParameter)->clientSocket, Buffer, sizeof(Buffer), 0);
        memcpy(Cache[last_cache].buffer, Buffer, strlen(Buffer));
        createAndWriteToFile(Buffer, Cache[last_cache].url);
        if (ret != SOCKET_ERROR)
        {
            printf("来自服务器\n代理服务器转发报文头为\n");
            char *headerPartPtr = strstr(Buffer, "\r\n\r\n");
            int length = headerPartPtr - Buffer + 4;
            char *headerPart = new char[length + 1];
            strncpy(headerPart, Buffer, length);
            headerPart[length] = '\0';
            printf("%s\n", headerPart);
            delete[] headerPart;
        }
    }
    // 错误处理
error:
    printf("error，关闭套接字\n\n");
    Sleep(200);
    closesocket(((ProxyParam *)lpParameter)->clientSocket);
    closesocket(((ProxyParam *)lpParameter)->serverSocket);
    free(lpParameter);
    _endthreadex(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("代理服务器正在启动\n");
    printf("初始化...\n");
    if (!InitWsa())
    {
        printf("WSA 初始化失败\n");
        return -1;
    }
    if (!InitSocket())
    {
        printf("代理服务器 socket 初始化失败\n");
        return -1;
    }
    printf("代理服务器正在运行，监听端口 %d\n", ProxyPort);
    SOCKET acceptSocket = INVALID_SOCKET; // 用于真正通信的套接字，初始化为无效套接字
    SOCKADDR_IN acceptAddr;               // 客户端端点信息
    ProxyParam *lpProxyParam;             // 代理服务器参数
    HANDLE hThread;                       // 线程句柄
    DWORD dwThreadID;                     // 线程ID

    // 代理服务器不断监听
    while (true)
    {
        acceptSocket = accept(ProxyServer, (SOCKADDR *)&acceptAddr, NULL);

        lpProxyParam = new ProxyParam;
        if (lpProxyParam == NULL)
        {
            continue;
        }
        lpProxyParam->clientSocket = acceptSocket;
        hThread = (HANDLE)_beginthreadex(NULL, 0,
                                         &ProxyThread, (LPVOID)lpProxyParam, 0, 0);
        CloseHandle(hThread);
        Sleep(200);
    }
    closesocket(ProxyServer);
    WSACleanup();
    return 0;
}