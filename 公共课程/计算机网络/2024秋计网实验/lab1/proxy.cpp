#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <process.h>
#include <string.h>
#include <tchar.h>
#pragma comment(lib, "Ws2_32.lib")

#define MAXSIZE 65507 //发送数据报文的最大长度
#define HTTP_PORT 80 //http 服务器端口

//Http 重要头部数据
struct HttpHeader {
    char method[8]; // POST 或者 GET，注意有些为 CONNECT，本实验暂不考虑
    char url[1024]; // 请求的 url
    char host[1024]; // 目标主机
    char cookie[1024 * 10]; //cookie
    HttpHeader() {
        ZeroMemory(this, sizeof(HttpHeader));
    }
};
// 代理服务器的发送套接字和接收套接字
struct ProxyParam {
    SOCKET clientSocket;
    SOCKET serverSocket;
};

//代理相关参数
SOCKET ProxyServer;                    // 代理服务器的套接字
sockaddr_in ProxyServerAddr;           // 代理服务器地址
const int ProxyPort = 8080;           // 代理服务器监听的端口

//由于新的连接都使用新线程进行处理，对线程的频繁的创建和销毁特别浪费资源
//可以使用线程池技术提高服务器效率
//const int ProxyThreadMaxNum = 20;
//HANDLE ProxyThreadHandle[ProxyThreadMaxNum] = {0};
//DWORD ProxyThreadDW[ProxyThreadMaxNum] = {0};

BOOL InitSocket();
void ParseHttpHead(char* buffer, HttpHeader* httpHeader);
BOOL ConnectToServer(SOCKET* serverSocket, char* host);
unsigned int __stdcall ProxyThread(LPVOID lpParameter);


void GetFileName(char* url, char* filename);
void GetDate(char* FileBuffer, char* date);
void NewMessage(char* buffer, char* date);
void SaveMessage(char* buffer, char* filename);
void ParseNewMessage(char* buffer, char* filename);

char invalid_website[25] = "http://www.hit.edu.cn/"; // 网站过滤
char restrict_host[25] = "127.0.0.1";                // 用户过滤
char fishing_src[25] = "http://news.hit.edu.cn/";   // 钓鱼网站原网址
char fishing_dest[25] = "http://jwts.hit.edu.cn/";   // 钓鱼网站目标网址
char fishing_dest_host[25] = "jwts.hit.edu.cn";      // 钓鱼目的地址主机名

int _tmain(int argc, _TCHAR* argv[])
{
    printf("代理服务器正在启动\n");
    printf("初始化...\n");
    if (!InitSocket()) {
        printf("socket 初始化失败\n");
        return -1;
    }
    printf("代理服务器正在运行，监听端口 %d\n", ProxyPort);
    SOCKET acceptSocket = INVALID_SOCKET;
    ProxyParam* lpProxyParam;
    HANDLE hThread;
    DWORD dwThreadID;
    SOCKADDR_IN clientAddr;
    int addr_len = sizeof(SOCKADDR_IN);

    //代理服务器不断监听
    while (true) {
        acceptSocket = accept(ProxyServer, (SOCKADDR*)&clientAddr, &addr_len); // 后两个参数用来记录客户端的地址和地址长度, 以存储用户地址

        // 屏蔽用户
        //if(!strcmp(restrict_host, inet_ntoa(clientAddr.sin_addr))){    // inet_ntoa：将ipv4网络地址转换为点分十进制地址
        //    printf("该用户访问受限\n\n");
        //    closesocket(acceptSocket);
        //    continue;
        //}
       //监测是否接收到了无效的套接字
        if (acceptSocket == INVALID_SOCKET) {
            printf("与客户端数据传送的套接字建立失败，错误代码为：%d\n", WSAGetLastError());
            return 0;
        }
        lpProxyParam = new ProxyParam;
        if (lpProxyParam == NULL) {
            printf("申请堆空间失败，关闭接收套接字\n");
            closesocket(acceptSocket);
            continue;
        }
        lpProxyParam->clientSocket = acceptSocket;
        hThread = (HANDLE)_beginthreadex(NULL, 0, &ProxyThread, (LPVOID)lpProxyParam, 0, 0);   // 创建线程
        if (hThread == NULL) {
            printf("新线程创建失败，关闭接收套接字\n");
            closesocket(acceptSocket);
            continue;
        }
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
BOOL InitSocket()
{
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
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
        printf("不能找到正确的 winsock 版本\n");
        WSACleanup();
        return FALSE;
    }
    ProxyServer = socket(AF_INET, SOCK_STREAM, 0);
    if (INVALID_SOCKET == ProxyServer) {
        printf("创建套接字失败，错误代码为：%d\n", WSAGetLastError());
        return FALSE;
    }
    ProxyServerAddr.sin_family = AF_INET;        // 使用IP协议簇
    ProxyServerAddr.sin_port = htons(ProxyPort);  // 本地z字节--->网络字节
    ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;    // 地址通配符
    if (bind(ProxyServer, (SOCKADDR*)&ProxyServerAddr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
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
unsigned int __stdcall ProxyThread(LPVOID lpParameter)
{
    char Buffer[MAXSIZE];       // 用于接收报文
    char* CacheBuffer;          // 用于解析报文
    char FileBuffer[MAXSIZE];   // 本地缓存
    char filename[300];         // 本地文件名
    char date[30];              // 缓存日期
    ZeroMemory(Buffer, MAXSIZE);
    ZeroMemory(FileBuffer, MAXSIZE);
    ZeroMemory(filename, 300);
    ZeroMemory(date, 30);
    SOCKADDR_IN clientAddr;
    int length = sizeof(SOCKADDR_IN);
    int recvSize;
    int ret;
    HttpHeader* httpHeader = new HttpHeader();
    bool hitcache = FALSE;
    FILE* fp;

    // 代理接收来自客户端的数据，并进行头部解析
    recvSize = recv(((ProxyParam*)lpParameter)->clientSocket, Buffer, MAXSIZE, 0);
    if (recvSize <= 0) {
        printf("接收客户端数据出错，线程关闭\n");
        goto error;
    }
    CacheBuffer = new char[recvSize + 1];
    ZeroMemory(CacheBuffer, recvSize + 1);
    memcpy(CacheBuffer, Buffer, recvSize);
    ParseHttpHead(CacheBuffer, httpHeader);
    delete CacheBuffer;

    // 屏蔽CONNECT报文
    if (strcmp(httpHeader->method, "CONNECT") == 0) {
        goto error;
    }

    // 网站过滤
    if (strstr(httpHeader->url, invalid_website) != NULL) {
        printf("\n===================================\n");
        printf("----------该网站已被屏蔽!----------\n");
        goto error;
    }

    // 钓鱼
    if (strstr(httpHeader->url, fishing_src) != NULL) {
        printf("\n========================================================================================\n");
        printf("------已从源网址：%s 转到 目的网址：%s ------\n\n", fishing_src, fishing_dest);
        memcpy(httpHeader->host, fishing_dest_host, strlen(fishing_dest_host) + 1);
        memcpy(httpHeader->url, fishing_dest, strlen(fishing_dest));
    }

    // 查询缓存是否命中（代理是否存在相应缓存）
    GetFileName(httpHeader->url, filename);//提取文件名
    if (!fopen_s(&fp, filename, "rb")) {    // 缓存存在，更改旧缓存的时间date
        fread(FileBuffer, sizeof(char), MAXSIZE, fp);
        fclose(fp);
        GetDate(FileBuffer, date);
        NewMessage(Buffer, date);
        hitcache = TRUE;
    }

    // 与目标服务器建立连接
    if (!ConnectToServer(&((ProxyParam*)lpParameter)->serverSocket, httpHeader->host)) {
        printf("代理未连接服务器，线程关闭\n");
        goto error;
    }
    printf("代理连接主机 %s 成功\n", httpHeader->host);

    //将客户端发送的 HTTP 数据报文直接转发给目标服务器（或是缓存命中更改后的报文）
    ret = send(((ProxyParam*)lpParameter)->serverSocket, Buffer, strlen(Buffer) + 1, 0);
    if (ret == -1) {
        printf("代理服转发报文给服务器失败，线程关闭\n");
    }

    //等待目标服务器返回数据
    recvSize = recv(((ProxyParam*)lpParameter)->serverSocket, Buffer, MAXSIZE, 0);
    if (recvSize <= 0) {
        printf("接收服务器数据出错，线程关闭\n");
        goto error;
    }

    // 再次查询代理是否有缓存
    if (!hitcache) {
        SaveMessage(Buffer, filename);  // 若没有命中，则缓存服务器返回的对象
    }
    else {
        ParseNewMessage(Buffer, filename);  // 如果命中，则解析首部信息，检查是否需要更新缓存
    }

    //将目标服务器返回的数据直接转发给客户端
    ret = send(((ProxyParam*)lpParameter)->clientSocket, Buffer, sizeof(Buffer), 0);
    if (ret == -1) {
        printf("代理转发报文给客户端失败，线程关闭\n");
    }

    //错误处理
error:
    Sleep(200);
    closesocket(((ProxyParam*)lpParameter)->clientSocket);
    closesocket(((ProxyParam*)lpParameter)->serverSocket);
    delete (ProxyParam*)lpParameter;
    delete httpHeader;
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
void ParseHttpHead(char* buffer, HttpHeader* httpHeader)
{
    char* p;
    char* ptr;
    const char* delim = "\r\n";
    p = strtok_s(buffer, delim, &ptr);//提取第一行
    if (p[0] == 'G') {// GET方式
        memcpy(httpHeader->method, "GET", 3);
        memcpy(httpHeader->url, &p[4], strlen(p) - 13);
    }
    else if (p[0] == 'P') {// POST方式
        memcpy(httpHeader->method, "POST", 4);
        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
    }
    else if (p[0] == 'C') {// CONNECT方法
        memcpy(httpHeader->method, "CONNECT", 8);
        memcpy(httpHeader->url, &p[9], strlen(p) - 18);
        return;
    }
    printf("\n%s\n", p);
    // printf("%s\n",httpHeader->url);
    p = strtok_s(NULL, delim, &ptr);
    while (p) {
        switch (p[0]) {
        case 'H'://Host
            memcpy(httpHeader->host, &p[6], strlen(p) - 6);
            break;
        case 'C'://Cookie
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
////void ParseHttpHead(char *buffer, HttpHeader * httpHeader) 
//{
//    char* p;
//    char* ptr;
//    const char* delim = "\r\n";
//    p = strtok_s(buffer, delim, &ptr);//提取第一行 
//    printf("%s\n", p);
//    if (p[0] == 'G') {//GET 方式 
//        memcpy(httpHeader->method, "GET", 3);
//        memcpy(httpHeader->url, &p[4], strlen(p) - 13);
//    }
//    else if (p[0] == 'P') {//POST 方式 
//        memcpy(httpHeader->method, "POST", 4);
//        memcpy(httpHeader->url, &p[5], strlen(p) - 14);
//    }
//    else if (p[0] == 'C') {// CONNECT方法
//        memcpy(httpHeader->method, "CONNECT", 8);
//        memcpy(httpHeader->url, &p[9], strlen(p) - 18);
//        return;
//    }
//    printf("\n%s\n", p);
//    //printf("%s\n", httpHeader->url);
//    p = strtok_s(NULL, delim, &ptr);
//    while (p) {
//        switch (p[0]) {
//        case 'H'://Host 
//            memcpy(httpHeader->host, &p[6], strlen(p) - 6);
//            break;
//        case 'C'://Cookie 
//            if (strlen(p) > 8) {
//                char header[8];
//                ZeroMemory(header, sizeof(header));
//                memcpy(header, p, 6);
//                if (!strcmp(header, "Cookie")) {
//                    memcpy(httpHeader->cookie, &p[8], strlen(p) - 8);
//                }
//            }
//            break;
//        default:
//            break;
//        }
//        p = strtok_s(NULL, delim, &ptr);
//    }
//}

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
    sockaddr_in serverAddr;  // 配置原服务器地址
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(HTTP_PORT);  // 本地->网络字节
    HOSTENT* hostent = gethostbyname(host);  // 域名->32位IP地址
    if (!hostent) {
        printf("通过域名获取IP地址失败\n");
        return FALSE;
    }
    in_addr Inaddr = *((in_addr*)*hostent->h_addr_list);
    serverAddr.sin_addr.s_addr = inet_addr(inet_ntoa(Inaddr));

    *serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (*serverSocket == INVALID_SOCKET) {
        printf("创建代理发送套接字失败，错误代码为： %d\n", WSAGetLastError());
        return FALSE;
    }
    if (connect(*serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        printf("代理连接失败，关闭代理发送套接字\n");
        closesocket(*serverSocket);
        return FALSE;
    }
    return TRUE;
}
//zty
// 提取请求报文中的文件名,将文件名中的空格和下划线进行替换处理
void GetFileName(char* url, char* filename)
{
    int i = 7; // 抛去http://
    while (url[i] != '\0') {
        if (url[i] == '/') {
            filename[i - 7] = ' ';
            i++;
        }
        else if (url[i] == '.') {
            filename[i - 7] = '_';
            i++;
        }
        else {
            filename[i - 7] = url[i];
            i++;
        }
    }
    filename[i - 8] = '\0';
}

// 提取请求报文中的日期(从给定的文件缓冲区 FileBuffer 中提取日期信息)
void GetDate(char* FileBuffer, char* date)
{
    char* p;
    char* ptr = NULL;
    const char* delim = "\r\n";
    const char* field = "Date";
    p = strtok_s(FileBuffer, delim, &ptr);//将文件缓冲区 FileBuffer 按照分隔符 \r\n 进行分割
    while (p) {
        if (strstr(p, field) != NULL) {  // 判断是否为子串
            memcpy(date, &p[6], strlen(p) - 6);//抛去 "Date: "
            break;
        }
        p = strtok_s(NULL, delim, &ptr);  // 不断分行
    }
}

// 更改请求报文（缓存已经命中，需要添加if-modified-since首部行）
void NewMessage(char* buffer, char* date)
{
    char* p;
    char* ptr = NULL;
    const char* delim = "\r\n";
    const char* field = "Host: ";
    const char* newfield = "If-Modified-Since: ";   // 插入到字段Host的下一行

    p = strstr(buffer, field);
    if (p) {
        while (*p != '\n') {
            p++;
        }
        p++;
    }//p 指向换行符后面的字符。
    else {
        printf("没有Host首部行, 程序关闭\n");
        exit(0);
    }
    // 找到插入位置
    char temp[MAXSIZE];  // 存储插入位置之后的所有临时变量
    ZeroMemory(temp, MAXSIZE);
    int i = 0;
    while (p[i] != '\0') {
        temp[i] = p[i];       // 缓存临时变量
        i++;
    }
    while (*newfield != '\0') {
        *(p++) = *(newfield++);   // 插入If-Modified-Since字段
    }
    while (*date != '\0') {
        *(p++) = *(date++);      // 插入日期
    }
    *(p++) = '\r';
    *(p++) = '\n';
    i = 0;
    while (temp[i] != '\0') {
        *(p++) = temp[i];        // 将临时变量添加回去
        i++;
    }
    printf("----------------------------------------代理请求报文------------------------------------\n");
    printf("%s", buffer);
}

// 保留缓存到本地
void SaveMessage(char* buffer, char* filename)
{
    char* p;
    char* ptr = NULL;
    const char* delim = "\r\n";
    char temp[MAXSIZE];
    ZeroMemory(temp, MAXSIZE);
    memcpy(temp, buffer, MAXSIZE);
    p = strtok_s(temp, delim, &ptr);//使用 strtok_s 函数来按行分割 temp 中的内容，并将第一行的指针赋值给 p
    char state[4];
    state[3] = '\0';
    memcpy(state, &p[9], 3); // p[9]是抛去"http/1.1 "

    if (!strcmp(state, "200")) { //状态码正常时缓存
        printf("代理服务器缓存完毕\n");
        FILE* fp;
        if (!fopen_s(&fp, filename, "wb")) { // 成功打开文件
            fwrite(buffer, sizeof(char), MAXSIZE, fp); // 按照最大格式写报文
            fclose(fp);
        }
        else {
            printf("文件打开失败\n");
        }
    }
    printf("state: %s\n\n", state);
}

// 解析服务器的响应报文（缓存已经命中，查询更新）
void ParseNewMessage(char* buffer, char* filename)
{
    char* p;
    char* ptr = NULL;
    const char* delim = "\r\n";
    char temp[MAXSIZE];
    ZeroMemory(temp, MAXSIZE);
    memcpy(temp, buffer, MAXSIZE);//将接收到的消息内容 buffer 复制到临时数组 temp 中
    p = strtok_s(temp, delim, &ptr);//将 temp 中的内容按照分隔符 \r\n 进行分割
    char state[4];//存储状态码
    state[3] = '\0';//将 state 数组的最后一个元素设为字符串结束符
    memcpy(state, &p[9], 3); // p[9]是抛去"http/1.1 "
    if (!strcmp(state, "304")) { // 状态码304, 无需缓存（304状态码表示资源自上次请求以来没有修改过。 
        //当浏览器再次请求同一个资源时，如果服务器返回304状态码，浏览器会直接使用缓存的版本，而不会向服务器发送请求。）
        printf("代理服务器已经缓存\n");
        ZeroMemory(buffer, MAXSIZE);
        FILE* fp;
        if (!fopen_s(&fp, filename, "rb")) { // 成功打开文件
            fread(buffer, sizeof(char), MAXSIZE, fp); //将文件内容读取到 buffer 中
            fclose(fp);
        }
    }
    else {
        printf("代理服务器重新缓存\n");
        SaveMessage(buffer, filename);//将消息内容缓存到指定文件中
    }
}