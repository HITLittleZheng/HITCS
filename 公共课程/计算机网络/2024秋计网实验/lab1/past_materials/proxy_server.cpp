//#include "stdafx.h"
#include <tchar.h>
#include <stdio.h>
#include <Windows.h>
#include <process.h>
#include <string.h>

//静态加入一个lib库文件,ws2_32.lib文件，提供了对以下网络相关API的支持，若使用其中的API，则应该将ws2_32.lib加入工程
#pragma comment(lib, "Ws2_32.lib")

#define MAXSIZE 65507 // 发送数据报文的最大长度
#define HTTP_PORT 80  // http 服务器端口

#define INVALID_WEBSITE "http://http.p2hp.com/"   //网站过滤
#define FISH_WEBSITE_FROM "http://http.p2hp.com/"  //钓鱼网站源网址
#define FISH_WEBSITE_TO "http://jwes.hit.edu.cn/"  //钓鱼网站目的网址
#define FISH_WEBSITE_HOST "jwes.hit.edu.cn"        //钓鱼目的地址的主机名

// Http 重要头部数据
struct HttpHeader
{
    char method[4];         // POST 或者 GET，注意有些为 CONNECT，本实验暂不考虑
    char url[1024];         // 请求的 url
    char host[1024];        // 目标主机
    char cookie[1024 * 10]; // cookie
    HttpHeader()
    {
        ZeroMemory(this, sizeof(HttpHeader));
    }
};

BOOL InitSocket();
void ParseHttpHead(char* buffer, HttpHeader* httpHeader);
BOOL ConnectToServer(SOCKET* serverSocket, char* host);
unsigned int __stdcall ProxyThread(LPVOID lpParameter);
void getfileDate(FILE* in, char* tempDate);//从文件中读取日期信息，并将日期信息存储在tempDate中
void sendnewHTTP(char* buffer, char* datestring);//发送一个新的HTTP请求，并将响应存储在buffer中
void makeFilename(char* url, char* filename);//根据URL生成文件名
void storefileCache(char* buffer, char* url);//将文件内容存储到缓存中
void checkfileCache(char* buffer, char* filename);//检查缓存中是否存在该文件，如果存在，则将文件内容返回


/*
代理相关参数:
    SOCKET：本质是一个unsigned int整数，是唯一的ID
    sockaddr_in：用来处理网络通信的地址。sockaddr_in用于socket定义和赋值；sockaddr用于函数参数
        short   sin_family;         地址族
        u_short sin_port;           16位TCP/UDP端口号
        struct  in_addr sin_addr;   32位IP地址
        char    sin_zero[8];        不使用
*/
SOCKET ProxyServer;
sockaddr_in ProxyServerAddr;
const int ProxyPort = 10240;

//缓存相关参数
boolean haveCache = FALSE;
boolean needCache = TRUE;

// 由于新的连接都使用新线程进行处理，对线程的频繁的创建和销毁特别浪费资源
// 可以使用线程池技术提高服务器效率
// const int ProxyThreadMaxNum = 20;
// HANDLE ProxyThreadHandle[ProxyThreadMaxNum] = {0};
// DWORD ProxyThreadDW[ProxyThreadMaxNum] = {0};

struct ProxyParam
{
    SOCKET clientSocket;
    SOCKET serverSocket;
};

int _tmain(int argc, _TCHAR* argv[])
{
    printf("代理服务器正在启动\n");
    printf("初始化...\n");
    if (!InitSocket())
    {
        printf("socket 初始化失败\n");
        return -1;
    }
    printf("代理服务器正在运行，监听端口 %d\n", ProxyPort);
    SOCKET acceptSocket = INVALID_SOCKET; //无效套接字-->初始化？
    ProxyParam* lpProxyParam;
    HANDLE hThread; //句柄：和对象一一对应的32位无符号整数值
    DWORD dwThreadID; //unsigned long

    // 代理服务器不断监听
    while (true)
    {
        //accept将客户端的信息绑定到一个socket上，也就是给客户端创建一个socket，通过返回值返回给我们客户端的socket
        acceptSocket = accept(ProxyServer, NULL, NULL);
        lpProxyParam = new ProxyParam;
        if (lpProxyParam == NULL)
        {
            continue;
        }
        lpProxyParam->clientSocket = acceptSocket;

        //_beginthreadex创建线程，第3、4个参数分别为线程执行函数、线程函数的参数
        hThread = (HANDLE)_beginthreadex(NULL, 0, &ProxyThread, (LPVOID)lpProxyParam, 0, 0);

        /* CloseHandle
            只是关闭了一个线程句柄对象，表示我不再使用该句柄，
            即不对这个句柄对应的线程做任何干预了，和结束线程没有一点关系
        */
        CloseHandle(hThread);

        //延迟 from <windows.h>
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
BOOL InitSocket()
{

    // 加载套接字库（必须）
    WORD wVersionRequested;
    WSADATA wsaData;
    // 套接字加载时错误提示
    int err;
    // 版本 2.2
    wVersionRequested = MAKEWORD(2, 2);
    // 加载 dll 文件 Scoket 库
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0)
    {
        // 找不到 winsock.dll
        printf("加载 winsock 失败，错误代码为: %d\n", WSAGetLastError());
        return FALSE;
    }
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
    {
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup();
        return FALSE;
    }

    /*socket() 函数：
        int af：地址族规范。当前支持的值为AF_INET或AF_INET6，这是IPv4和IPv6的Internet地址族格式
        int type：新套接字的类型规范。SOCK_STREAM 1是一种套接字类型，可通过OOB数据传输机制提供
                  顺序的，可靠的，双向的，基于连接的字节流
        int protocol：协议。值为0，则调用者不希望指定协议，服务提供商将选择要使用的协议
    */
    ProxyServer = socket(AF_INET, SOCK_STREAM, 0);

    if (INVALID_SOCKET == ProxyServer)
    {
        printf("创建套接字失败，错误代码为： %d\n", WSAGetLastError());
        return FALSE;
    }

    ProxyServerAddr.sin_family = AF_INET;
    ProxyServerAddr.sin_port = htons(ProxyPort);    //将一个无符号短整型数值转换为TCP/IP网络字节序，即大端模式(big-endian)
    //ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;  //INADDR_ANY == 0.0.0.0，表示本机的所有IP
    ProxyServerAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");  //仅限本机用户可访问


    //绑定套接字与网络地址
    if (bind(ProxyServer, (SOCKADDR*)&ProxyServerAddr, sizeof(SOCKADDR)) == SOCKET_ERROR)
    {
        printf("绑定套接字失败\n");
        return FALSE;
    }

    //监听套接字
    if (listen(ProxyServer, SOMAXCONN) == SOCKET_ERROR)
    {
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
// 线程的生命周期就是线程函数从开始执行到线程结束
// //************************************
unsigned int __stdcall ProxyThread(LPVOID lpParameter)
{
    //定义缓存变量
    char filename[100] = { 0 };
    _Post_ _Notnull_ FILE* in;//定义了一个文件指针，用于指向输入文件。_Post_ _Notnull_是一个宏，表示在函数返回之后，in指针必须不为空
    char* DateBuffer;//指向日期缓冲区。日期缓冲区用于存储从网络数据包中解析出的日期值
    char date_str[30];  //定义了一个字符数组，用于存储日期字符串。日期字符串通常由年、月、日组成，例如 "2021-08-01"
    FILE* fp;

    char Buffer[MAXSIZE];
    char* CacheBuffer;
    ZeroMemory(Buffer, MAXSIZE);
    SOCKADDR_IN clientAddr;
    int length = sizeof(SOCKADDR_IN);
    int recvSize;
    int ret;
    recvSize = recv(((ProxyParam*)lpParameter)->clientSocket, Buffer, MAXSIZE, 0);
    HttpHeader* httpHeader = new HttpHeader();
    CacheBuffer = new char[recvSize + 1];

    //goto语句不能跳过实例化（局部变量定义），把实例化移到函数开头就可以了
    if (recvSize <= 0)
    {
        goto error;
    }
    
    ZeroMemory(CacheBuffer, recvSize + 1);
    memcpy(CacheBuffer, Buffer, recvSize);  //由src指向地址为起始地址的连续n个字节的数据复制到以destin指向地址为起始地址的空间内
    //解析httpheader
    ParseHttpHead(CacheBuffer, httpHeader); 

    ZeroMemory(date_str, 30);
    printf("httpHeader->url : %s\n", httpHeader->url);
    makeFilename(httpHeader->url, filename);
    //printf("filename是 %s\n", filename);
    if ((fopen_s(&in, filename, "r")) == 0)
    {
        printf("\n有缓存\n");

        getfileDate(in, date_str);//得到本地缓存文件中的日期date_str
        fclose(in);
        //printf("date_str:%s\n", date_str);
        sendnewHTTP(Buffer, date_str);
        //向服务器发送一个请求，该请求需要增加 “If-Modified-Since” 字段
        //服务器通过对比时间来判断缓存是否过期
        haveCache = TRUE;
    }

    delete CacheBuffer; 
    //printf("test\n");

    //在发送报文前进行拦截
    //if (strcmp(httpHeader->url,INVALID_WEBSITE)==0)
    //{
    //    printf("************************************\n");
    //    printf("----------该网站已被屏蔽------------\n");
    //    printf("************************************\n");
    //    goto error;
    //}

    ////网站引导
    if (strstr(httpHeader->url, FISH_WEBSITE_FROM) != NULL) {
        printf("\n=====================================\n\n");
        printf("-------------已从源网址：%s 转到 目的网址 ：%s ----------------\n", FISH_WEBSITE_FROM, FISH_WEBSITE_TO);
        memcpy(httpHeader->host, FISH_WEBSITE_HOST, strlen(FISH_WEBSITE_HOST) + 1);
        memcpy(httpHeader->url, FISH_WEBSITE_TO, strlen(FISH_WEBSITE_TO));
     }


    if (!ConnectToServer(&((ProxyParam*)lpParameter)->serverSocket, httpHeader->host))
    {
        //printf("error\n");
        goto error;
    }
    printf("代理连接主机 %s 成功\n", httpHeader->host);
    //printf("test");
   
    // 将客户端发送的 HTTP 数据报文直接转发给目标服务器
    ret = send(((ProxyParam*)lpParameter)->serverSocket, Buffer, strlen(Buffer) + 1, 0);
    
    // 等待目标服务器返回数据
    recvSize = recv(((ProxyParam*)lpParameter)->serverSocket, Buffer, MAXSIZE, 0);
    if (recvSize <= 0)
    {
        goto error;
    }
    if (haveCache == true) {
        checkfileCache(Buffer, httpHeader->url);
    }
    if (needCache == true) {
        storefileCache(Buffer, httpHeader->url);
    }

    // 将目标服务器返回的数据直接转发给客户端
    ret = send(((ProxyParam*)lpParameter)->clientSocket, Buffer, sizeof(Buffer), 0);


// 错误处理
error:
    printf("关闭套接字\n");
    Sleep(200);
    closesocket(((ProxyParam*)lpParameter)->clientSocket);
    closesocket(((ProxyParam*)lpParameter)->serverSocket);
    delete lpParameter;
    _endthreadex(0);//中止线程
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
void ParseHttpHead(char* buffer, HttpHeader* httpHeader)
{
    char* p;
    char* ptr;
    const char* delim = "\r\n";
    p = strtok_s(buffer, delim, &ptr); // 提取第一行
    printf("%s\n", p);
    if (p[0] == 'G')
    { // GET 方式
        memcpy(httpHeader->method, "GET", 3);
        memcpy(httpHeader->url, &p[4], strlen(p) - 13);
    }
    else if (p[0] == 'P')
    { // POST 方式
        memcpy(httpHeader->method, "POST", 4);
        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
    }
    printf("正在访问url：%s\n", httpHeader->url);
    p = strtok_s(NULL, delim, &ptr);    //提取第二行
    while (p)
    {
        switch (p[0])
        {
        case 'H': // Host
            memcpy(httpHeader->host, &p[6], strlen(p) - 6); //将Host复制到httpHeader->host中
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
        p = strtok_s(NULL, delim, &ptr); //读取下一行
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
BOOL ConnectToServer(SOCKET* serverSocket, char* host)
{
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(HTTP_PORT);
    HOSTENT* hostent = gethostbyname(host);
    if (!hostent)
    {
        //printf("error_hostent\n");
        return FALSE;
    }
    in_addr Inaddr = *((in_addr*)*hostent->h_addr_list);
    serverAddr.sin_addr.s_addr = inet_addr(inet_ntoa(Inaddr));
    *serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (*serverSocket == INVALID_SOCKET)
    {
        //printf("error_serverSocket\n");
        return FALSE;
    }
    if (connect(*serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        //printf("error_connect\n");
        closesocket(*serverSocket);
        return FALSE;
    }
    return TRUE;
}

//访问本地文件，获取本地缓存中的日期
void getfileDate(FILE* in, char* tempDate)
{
    char field[5] = "Date";
    //ptr，用于存储strtok_s函数的返回值
    char* p, * ptr, temp[5];//p，用于存储从文件中读取的字符串

    char buffer[MAXSIZE];//存储从文件中读取的字符串
    ZeroMemory(buffer, MAXSIZE);
    fread(buffer, sizeof(char), MAXSIZE, in);//从文件中读取字符串，并将其存储在buffer数组中
    const char* delim = "\r\n";//换行符
    ZeroMemory(temp, 5);
    p = strtok_s(buffer, delim, &ptr);//使用strtok_s函数从buffer数组按行分割字符串，并将第一行存储在p中
    //printf("p = %s\n", p);
    int len = strlen(field) + 2;
    while (p) //如果p指针指向的字符串包含"Date"字符串，则使用memcpy函数将日期信息复制到tempDate数组中，并返回。
        //如果p指针指向的字符串不包含"Date"字符串，则继续遍历下一个字符串
    {
        if (strstr(p, field) != NULL) {//调用strstr后指针会指向匹配剩余的第一个字符
            memcpy(tempDate, &p[len], strlen(p) - len);
            return;
        }
        p = strtok_s(NULL, delim, &ptr);
    }
}

//改造HTTP请求报文
void sendnewHTTP(char* buffer, char* datestring) {
    const char* field = "Host";
    const char* newfield = "If-Modified-Since: ";//分别用于表示请求报文段中的Host字段和要插入的新字段
    //const char *delim = "\r\n";
    char temp[MAXSIZE];//存储插入新字段后的请求报文
    ZeroMemory(temp, MAXSIZE);
    char* pos = strstr(buffer, field);//获取请求报文段中Host后的部分信息
    int i = 0;
    for (i = 0; i < strlen(pos); i++) {
        temp[i] = pos[i];//将pos复制给temp
    }
    *pos = '\0';
    //将pos指针指向Host字段后的第一个字符，然后遍历新字段，将新字段中的每个字符插入到pos指针指向的位置
    while (*newfield != '\0') {  //插入If-Modified-Since字段
        *pos++ = *newfield++;
    }
    while (*datestring != '\0') {//插入对象文件的最新被修改时间
        *pos++ = *datestring++;
    }
    *pos++ = '\r';//符合报文格式
    *pos++ = '\n';
    for (i = 0; i < strlen(temp); i++)//将原始请求报文段中的剩余字符复制到新字段之后
    {
        *pos++ = temp[i];
    }
}

//根据url构造文件名
void makeFilename(char* url, char* filename) {
    while (*url != '\0') {
        if ('a' <= *url && *url <= 'z') {
            *filename++ = *url;//如果当前字符在'a'到'z'的范围内，将其复制到文件名字符串中
        }
        url++;
    }
    strcat_s(filename, strlen(filename) + 9, ".txt");
}

//检测服务器返回的状态码，如果是200则把数据进行本地更新缓存
void storefileCache(char* buffer, char* url) {
    char* p, * ptr, tempBuffer[MAXSIZE + 1];

    const char* delim = "\r\n";
    ZeroMemory(tempBuffer, MAXSIZE + 1);
    memcpy(tempBuffer, buffer, strlen(buffer));
    p = strtok_s(tempBuffer, delim, &ptr);//提取第一行

    if (strstr(tempBuffer, "200") != NULL) {  //状态码是200时缓存
        char filename[100] = { 0 };
        makeFilename(url, filename);
        printf("filename : %s\n", filename);
        FILE* out;
        fopen_s(&out, filename, "w+");
        fwrite(buffer, sizeof(char), strlen(buffer), out);//使用fopen_s函数以写入模式打开文件，并将响应内容写入文件
        fclose(out);
        printf("\n===================更新缓存ok==================\n");
    }
}

//检测服务器返回的状态码，如果是304则从本地获取缓存进行转发，否则需要更新缓存
void checkfileCache(char* buffer, char* filename)
{
    char* p, * ptr, tempBuffer[MAXSIZE + 1];
    const char* delim = "\r\n";
    ZeroMemory(tempBuffer, MAXSIZE + 1);
    memcpy(tempBuffer, buffer, strlen(buffer));//将buffer复制到tempBuffer中
    p = strtok_s(tempBuffer, delim, &ptr);//提取状态码所在行
    //主机返回的报文中的状态码为304时返回已缓存的内容
    if (strstr(p, "304") != NULL) {
        printf("\n=================从本机获得缓存====================\n");
        ZeroMemory(buffer, strlen(buffer));
        FILE* in = NULL;
        if ((fopen_s(&in, filename, "r")) == 0) {
            fread(buffer, sizeof(char), MAXSIZE, in);//使用fopen_s函数以读取模式打开文件，并将文件内容存储在buffer数组中
            fclose(in);
        }
        needCache = FALSE;
    }
}
