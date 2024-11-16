//
// Created by 11620 on 2024/9/24.
//
#include <stdio.h>
#include <Windows.h>
#include <process.h>
#include <string.h>
#pragma comment(lib,"Ws2_32.lib")
#define MAXSIZE 65507 //发送数据报文的最大长度
#define HTTP_PORT 80 //http 服务器端口
#define DATELENGTH 50 //时间字节数
#define CACHE_NUM 50  //定义最大缓存数量

//Http 重要头部数据
struct HttpHeader {
    char method[4]; // POST 或者 GET，注意有些为 CONNECT，本实验暂不考虑
    char url[1024]; // 请求的 url
    char host[1024]; // 目标主机
    char cookie[1024 * 10]; //cookie
    HttpHeader() {
        ZeroMemory(this, sizeof(HttpHeader));
    }
};

//因为不做外部存储，所以为了节省空间，cache存储的时候
//去掉Http头部信息中的cookie
struct cacheHttpHead {
    char method[4]; // POST 或者 GET，注意有些为 CONNECT，本实验暂不考虑
    char url[1024]; // 请求的 url
    char host[1024]; // 目标主机
    cacheHttpHead() {
        ZeroMemory(this, sizeof(cacheHttpHead));
    }
};

//代理服务器缓存技术
struct CACHE {
    cacheHttpHead httpHead;
    char buffer[MAXSIZE]; //储存报文返回内容
    char date[DATELENGTH]; //缓存内容的最后修改时间
    CACHE() {
        ZeroMemory(this->buffer, MAXSIZE);
        ZeroMemory(this->date, DATELENGTH);
    }
};

CACHE cache[CACHE_NUM]; //缓存地址
int cache_index = 0; //记录当前应该将缓存放在哪个位置

BOOL InitSocket();

void ParseHttpHead(char *buffer, HttpHeader *httpHeader);

BOOL ConnectToServer(SOCKET *serverSocket, char *host);

unsigned int __stdcall ProxyThread(LPVOID lpParameter);

int isInCache(CACHE *cache, HttpHeader httpHeader); //寻找缓存中是否存在，如果存在返回index，不存在返回-1
BOOL httpEqual(cacheHttpHead http1, HttpHeader http2); //判断两个http报文是否相同,主要用来判断缓存和报文是否相同
void changeHTTP(char *buffer, char *date); //用于修改HTTP报文
//代理相关参数
SOCKET ProxyServer;
sockaddr_in ProxyServerAddr;
const int ProxyPort = 10240;
//由于新的连接都使用新线程进行处理，对线程的频繁的创建和销毁特别浪费资源
//可以使用线程池技术提高服务器效率
//const int ProxyThreadMaxNum = 20;
//HANDLE ProxyThreadHandle[ProxyThreadMaxNum] = {0};
//DWORD ProxyThreadDW[ProxyThreadMaxNum] = {0};
struct ProxyParam {
    SOCKET clientSocket;
    SOCKET serverSocket;
};

//选做功能参数定义
bool button = true; //取true的时候表示开始运行选做功能
//禁止访问网站
char *invalid_website[10] = {"http://www.hit.edu.cn"};
const int invalid_website_num = 1; //有多少个禁止网站
//钓鱼网站
char *fishing_src = "http://today.hit.edu.cn"; //钓鱼网站原网址
char *fishing_dest = "http://jwes.hit.edu.cn"; //钓鱼网站目标网址
char *fishing_dest_host = "jwts.hit.edu.cn"; //钓鱼目的地址主机名
//限制访问用户
char *restrict_host[10] = {"127.0.0.1"};

int main(int argc, char *argv[]) {
    printf("代理服务器正在启动\n");
    printf("初始化...\n");
    if (!InitSocket()) {
        printf("socket 初始化失败\n");
        return -1;
    }
    printf("代理服务器正在运行，监听端口 %d\n", ProxyPort);
    SOCKET acceptSocket = INVALID_SOCKET;
    ProxyParam *lpProxyParam;
    HANDLE hThread;
    DWORD dwThreadID;
    sockaddr_in addr_in;
    int addr_len = sizeof(SOCKADDR);
    //代理服务器不断监听
    while (true) {
        acceptSocket = accept(ProxyServer, (SOCKADDR *) &addr_in, &(addr_len));
        lpProxyParam = new ProxyParam;
        if (lpProxyParam == NULL) {
            continue;
        }
        //受限用户,与列表中匹配上的都无法访问
        if (strcmp(restrict_host[0], inet_ntoa(addr_in.sin_addr)) && button) //注意比较之前将网络二进制的数字转换成网络地址
        {
            printf("该用户访问受限\n");
            continue;
        }
        lpProxyParam->clientSocket = acceptSocket;
        hThread = (HANDLE) _beginthreadex(NULL, 0,
                                          &ProxyThread, (LPVOID) lpProxyParam, 0, 0);
        CloseHandle(hThread);
        Sleep(200);
    }
    closesocket(ProxyServer);
    WSACleanup();
    return 0;
}

//************************************
// Method: InitSocket
// FullName: InitSocket
// Access: public
// Returns: BOOL
// Qualifier: 初始化套接字
//************************************
BOOL InitSocket() {
    //加载套接字库（必须）
    WORD wVersionRequested;
    WSADATA wsaData;
    //套接字加载时错误提示
    int err;
    //版本 2.2
    wVersionRequested = MAKEWORD(2, 2);
    //加载 dll 文件 Scoket 库
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0) {
        //找不到 winsock.dll
        printf("加载 winsock 失败，错误代码为: %d\n", WSAGetLastError());
        return FALSE;
    }
    //if中的语句主要用于比对是否是2.2版本
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup();
        return FALSE;
    }
    //创建的socket文件描述符基于IPV4，TCP
    ProxyServer = socket(AF_INET, SOCK_STREAM, 0);
    if (INVALID_SOCKET == ProxyServer) {
        printf("创建套接字失败，错误代码为： %d\n", WSAGetLastError());
        return FALSE;
    }
    ProxyServerAddr.sin_family = AF_INET;
    ProxyServerAddr.sin_port = htons(ProxyPort); //整型变量从主机字节顺序转变成网络字节顺序,转换为大端法
    ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY; //泛指本机也就是表示本机的所有IP，多网卡的情况下，这个就表示所有网卡ip地址的意思
    if (bind(ProxyServer, (SOCKADDR *) &ProxyServerAddr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
        printf("绑定套接字失败\n");
        return FALSE;
    }
    if (listen(ProxyServer, SOMAXCONN) == SOCKET_ERROR) {
        printf("监听端口%d 失败", ProxyPort);
        return FALSE;
    }
    return TRUE;
}

//************************************
// Method: ProxyThread
// FullName: ProxyThread
// Access: public
// Returns: unsigned int __stdcall
// Qualifier: 线程执行函数
// Parameter: LPVOID lpParameter
//************************************
unsigned int __stdcall ProxyThread(LPVOID lpParameter) {
    char Buffer[MAXSIZE];
    char *CacheBuffer;
    ZeroMemory(Buffer, MAXSIZE);
    SOCKADDR_IN clientAddr;
    int length = sizeof(SOCKADDR_IN);
    int recvSize;
    int ret;
    recvSize = recv(((ProxyParam
                        *) lpParameter)->clientSocket, Buffer, MAXSIZE, 0); //接收到报文
    if (recvSize <= 0) {
        goto error;
    }
    HttpHeader *httpHeader = new HttpHeader();
    CacheBuffer = new char[recvSize + 1];
    ZeroMemory(CacheBuffer, recvSize + 1);
    memcpy(CacheBuffer, Buffer, recvSize);
    //处理HTTP头部
    ParseHttpHead(CacheBuffer, httpHeader);
    //处理禁止访问网站
    if (strstr(httpHeader->url, invalid_website[0]) != NULL && button) {
        printf("\n=====================\n");
        printf("--------该网站已被屏蔽!----------\n");
        goto error;
    }
    //处理钓鱼网站
    if (strstr(httpHeader->url, fishing_src) != NULL && button) {
        printf("\n=====================\n");
        printf("-------------已从源网址：%s 转到 目的网址 ：%s ----------------\n", fishing_src, fishing_dest);
        //修改HTTP报文
        memcpy(httpHeader->host, fishing_dest_host, strlen(fishing_dest_host) + 1);
        memcpy(httpHeader->url, fishing_dest, strlen(fishing_dest));
    }
    delete CacheBuffer;
    //连接目标主机
    if (!ConnectToServer(&((ProxyParam
                             *) lpParameter)->serverSocket, httpHeader->host)) {
        goto error;
    }
    printf("代理连接主机 %s 成功\n", httpHeader->host);

    int index = isInCache(cache, *httpHeader);
    //如果在缓存中存在
    if (index > -1) {
        char *cacheBuffer;
        char Buf[MAXSIZE];
        ZeroMemory(Buf, MAXSIZE);
        memcpy(Buf, Buffer, recvSize);
        //插入"If-Modified-Since: "
        changeHTTP(Buf, cache[index].date);
        printf("-------------------请求报文------------------------\n%s\n", Buf);
        ret = send(((ProxyParam
                       *) lpParameter)->serverSocket, Buf, strlen(Buf) + 1, 0);
        recvSize = recv(((ProxyParam
                            *) lpParameter)->serverSocket, Buf, MAXSIZE, 0);
        printf("------------------Server返回报文-------------------\n%s\n", Buf);
        if (recvSize <= 0) {
            goto error;
        }
        char *No_Modified = "304";
        //没有改变，直接返回cache中的内容
        if (!memcmp(&Buf[9], No_Modified, strlen(No_Modified))) {
            ret = send(((ProxyParam
                           *) lpParameter)->clientSocket, cache[index].buffer, strlen(cache[index].buffer) + 1, 0);
            printf("将cache中的缓存返回客户端\n");
            printf("============================\n");
            goto error;
        }
    }
    //将客户端发送的 HTTP 数据报文直接转发给目标服务器
    ret = send(((ProxyParam *) lpParameter)->serverSocket, Buffer, strlen(Buffer)
                                                                   + 1, 0);
    //等待目标服务器返回数据
    recvSize = recv(((ProxyParam
                        *) lpParameter)->serverSocket, Buffer, MAXSIZE, 0);
    if (recvSize <= 0) {
        goto error;
    }
    //以下部分将返回报文加入缓存
    //从服务器返回报文中解析时间
    char *cacheBuffer2 = new char[MAXSIZE];
    ZeroMemory(cacheBuffer2, MAXSIZE);
    memcpy(cacheBuffer2, Buffer, MAXSIZE);
    char *delim = "\r\n";
    char date[DATELENGTH];
    char *nextStr;
    ZeroMemory(date, DATELENGTH);
    char *p = strtok_s(cacheBuffer2, delim, &nextStr);
    bool flag = false; //表示是否含有修改时间报文
    //不断分行，直到分出具有修改时间的那一行
    while (p) {
        if (p[0] == 'L') //找到Last-Modified:那一行
        {
            if (strlen(p) > 15) {
                char header[15];
                ZeroMemory(header, sizeof(header));
                memcpy(header, p, 14);
                if (!(strcmp(header, "Last-Modified:"))) {
                    memcpy(date, &p[15], strlen(p) - 15);
                    flag = true;
                    break;
                }
            }
        }
        p = strtok_s(NULL, delim, &nextStr);
    }
    if (flag) {
        if (index > -1) //说明已经有内容存在，只要改一下时间和内容
        {
            memcpy(&(cache[index].buffer), Buffer, strlen(Buffer));
            memcpy(&(cache[index].date), date, strlen(date));
        } else //第一次访问，需要完全缓存
        {
            memcpy(&(cache[cache_index % CACHE_NUM].httpHead.host), httpHeader->host, strlen(httpHeader->host));
            memcpy(&(cache[cache_index % CACHE_NUM].httpHead.method), httpHeader->method, strlen(httpHeader->method));
            memcpy(&(cache[cache_index % CACHE_NUM].httpHead.url), httpHeader->url, strlen(httpHeader->url));
            memcpy(&(cache[cache_index % CACHE_NUM].buffer), Buffer, strlen(Buffer));
            memcpy(&(cache[cache_index % CACHE_NUM].date), date, strlen(date));
            cache_index++;
        }
    }
    //将目标服务器返回的数据直接转发给客户端
    ret = send(((ProxyParam
                   *) lpParameter)->clientSocket, Buffer, sizeof(Buffer), 0);
    //错误处理
error:
    printf("关闭套接字\n");
    Sleep(200);
    closesocket(((ProxyParam *) lpParameter)->clientSocket);
    closesocket(((ProxyParam *) lpParameter)->serverSocket);
    delete lpParameter;
    _endthreadex(0);
    return 0;
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
void ParseHttpHead(char *buffer, HttpHeader *httpHeader) {
    char *p;
    char *ptr;
    const char *delim = "\r\n";
    p = strtok_s(buffer, delim, &ptr); //提取第一行
    printf("%s\n", p);
    if (p[0] == 'G') {
        //GET 方式
        memcpy(httpHeader->method, "GET", 3);
        memcpy(httpHeader->url, &p[4], strlen(p) - 13);
    } else if (p[0] == 'P') {
        //POST 方式
        memcpy(httpHeader->method, "POST", 4);
        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
    }
    printf("%s\n", httpHeader->url);
    p = strtok_s(NULL, delim, &ptr);
    while (p) {
        switch (p[0]) {
            case 'H': //Host
                memcpy(httpHeader->host, &p[6], strlen(p) - 6);
                break;
            case 'C': //Cookie
                if (strlen(p) > 8) {
                    char header[8];
                    ZeroMemory(header, sizeof(header));
                    memcpy(header, p, 6);
                    if (!strcmp(header, "Cookie")) {
                        memcpy(httpHeader->cookie, &p[8], strlen(p) - 8);
                    }
                }
                break;
            default:
                break;
        }
        p = strtok_s(NULL, delim, &ptr);
    }
}

//************************************
// Method: ConnectToServer
// FullName: ConnectToServer
// Access: public
// Returns: BOOL
// Qualifier: 根据主机创建目标服务器套接字，并连接
// Parameter: SOCKET * serverSocket
// Parameter: char * host
//************************************
BOOL ConnectToServer(SOCKET *serverSocket, char *host) {
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(HTTP_PORT);
    HOSTENT *hostent = gethostbyname(host);
    if (!hostent) {
        return FALSE;
    }
    in_addr Inaddr = *((in_addr *) *hostent->h_addr_list);
    serverAddr.sin_addr.s_addr = inet_addr(inet_ntoa(Inaddr)); //将一个将网络地址转换成一个长整数型数
    *serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (*serverSocket == INVALID_SOCKET) {
        return FALSE;
    }
    if (connect(*serverSocket, (SOCKADDR *) &serverAddr, sizeof(serverAddr))
        == SOCKET_ERROR) {
        closesocket(*serverSocket);
        return FALSE;
    }
    return TRUE;
}

BOOL httpEqual(cacheHttpHead http1, HttpHeader http2) {
    if (strcmp(http1.method, http2.method))return false;
    if (strcmp(http1.host, http2.host))return false;
    if (strcmp(http1.url, http2.url))return false;
    return true;
}

int isInCache(CACHE *cache, HttpHeader httpHeader) {
    int index = 0;
    for (; index < CACHE_NUM; index++) {
        if (httpEqual(cache[index].httpHead, httpHeader))return index;
    }
    return -1;
}

void changeHTTP(char *buffer, char *date) {
    //此函数在HTTP中间插入"If-Modified-Since: "
    const char *strHost = "Host";
    const char *inputStr = "If-Modified-Since: ";
    char temp[MAXSIZE];
    ZeroMemory(temp, MAXSIZE);
    char *pos = strstr(buffer, strHost); //找到Host位置
    int i = 0;
    //将host与之后的部分写入temp
    for (i = 0; i < strlen(pos); i++) {
        temp[i] = pos[i];
    }
    *pos = '\0';
    while (*inputStr != '\0') {
        //插入If-Modified-Since字段
        *pos++ = *inputStr++;
    }
    while (*date != '\0') {
        *pos++ = *date++;
    }
    *pos++ = '\r';
    *pos++ = '\n';
    //将host之后的字段复制到buffer中
    for (i = 0; i < strlen(temp); i++) {
        *pos++ = temp[i];
    }
}
