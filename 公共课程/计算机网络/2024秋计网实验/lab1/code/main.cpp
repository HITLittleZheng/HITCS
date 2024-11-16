#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <Windows.h>
#include <process.h>
#include <string.h>
#include <tchar.h>
#include <fstream>
#include<string>
#include<iostream>

using namespace std;

#pragma comment(lib,"Ws2_32.lib")
#define MAXSIZE 65507 //发送数据报文的最大长度
#define HTTP_PORT 80 //http 服务器端口

//用于存储 HTTP 请求头的信息
struct HttpHeader {
	char method[4]; // 存储 HTTP 方法，POST 或者 GET，注意有些为 CONNECT，本实验暂不考虑
	char url[1024];  // 请求的 url
	char host[1024]; // 目标主机
	char cookie[1024 * 10]; //cookie
	HttpHeader() {// 构造函数，初始化结构体的成员变量
		ZeroMemory(this, sizeof(HttpHeader));// 将结构体的内存空间清零
	}
};

//用户过滤
char ForbiddenIP[1024][17];// 存储被禁止访问的 IP 地址列表
int IPnum = 0;// 记录被禁止访问的 IP 地址的数量

//钓鱼网站
char fishUrl[1024][1024];// 存储钓鱼网站的 URL 列表
int fishUrlnum = 0;// 记录钓鱼网站的 URL 的数量

BOOL InitSocket();// 初始化套接字库和代理服务器套接字
void ParseHttpHead(char* buffer, HttpHeader* httpHeader);// 解析 HTTP 请求头并存储到结构体中
int ParseCacheHttpHead(char* buffer, HttpHeader* httpHeader);// 解析缓存中的 HTTP 响应头并存储到结构体中
BOOL ConnectToServer(SOCKET* serverSocket, char* host);// 连接到目标服务器并返回服务器套接字
unsigned int __stdcall ProxyThread(LPVOID lpParameter);// 线程函数声明，处理每个客户端请求的代理线程

BOOL ParseDate(char* buffer, char* field, char* tempDate);// 从 HTTP 响应头中解析日期字段
void makeNewHTTP(char* buffer, char* value);// 修改 HTTP 请求头中的主机名和 URL
void makeFilename(char* url, char* filename); // 根据 URL 生成缓存文件名
void makeCache(char* buffer, char* url);// 将 HTTP 响应内容写入缓存文件中
void getCache(char* buffer, char* filename);// 从缓存文件中读取 HTTP 响应内容

//缓存相关参数
BOOL haveCache = FALSE;// 标记是否有缓存文件可用
BOOL needCache = TRUE;// 标记是否需要缓存文件
char* strArr[100];//存储 HTTP 响应头中的各个字段

bool ForbiddenToConnect(char* httpheader);// 判断是否禁止访问某个网站
bool GotoFalseWebsite(char* url);// 判断是否需要跳转到钓鱼网站
void ParseCache(char* buffer, char* status, char* last_modified);// 解析缓存文件中的 HTTP 状态码和最后修改日期
bool UserIsForbidden(char* userID);  //用户过滤，判断是否禁止某个用户访问代理服务器

//代理相关参数
SOCKET ProxyServer;// 表示代理服务器的套接字
sockaddr_in ProxyServerAddr;// 表示代理服务器的地址信息
const int ProxyPort = 10240;// 表示代理服务器的监听端口

struct ProxyParam {// 传递给代理线程的参数
	SOCKET clientSocket;// 表示客户端的套接字
	SOCKET serverSocket;// 表示目标服务器的套接字
};

int _tmain(int argc, _TCHAR* argv[])// 参数分别表示命令行参数的个数和值
{
	printf("代理服务器正在启动\n");
	printf("初始化...\n");
	if (!InitSocket()) {
		printf("socket 初始化失败\n");
		return -1;
	}
	printf("代理服务器正在运行，监听端口 %d\n", ProxyPort);
	SOCKET acceptSocket = INVALID_SOCKET;// 表示接受客户端连接的临时套接字
	SOCKADDR_IN acceptAddr;// 表示客户端的地址信息
	ProxyParam* lpProxyParam;// 指向传递给代理线程的参数
	HANDLE hThread;// 表示代理线程的句柄
	DWORD dwThreadID;// 表示代理线程的 ID
	//代理服务器不断监听
	char client_IP[16];// 存储客户端的 IP 地址字符串
	//设置禁用IP
	memcpy(ForbiddenIP[IPnum++], "127.0.0.1", 10);
	//设置访问哪些网站会被重定向到钓鱼网站
	memcpy(fishUrl[fishUrlnum++], "http://pku.edu.cn/", 18);

	while (true) {

		/* 接受代理服务器套接字上的客户端连接请求，
		并返回一个新的套接字来与客户端通信，同时将客户端的地址信息存储到 acceptAddr 结构体中*/
		acceptSocket = accept(ProxyServer, (SOCKADDR*)&acceptAddr, NULL);


		//禁用用户IP访问
		int ff = sizeof(acceptAddr);
		//接受代理服务器套接字上的客户端连接请求，并返回一个新的套接字来与客户端通信，
		acceptSocket = accept(ProxyServer, (SOCKADDR*)&acceptAddr, &(ff));
		printf("获取用户IP地址：%s\n",inet_ntoa(acceptAddr.sin_addr));
		memcpy(client_IP, inet_ntoa(acceptAddr.sin_addr), 16);// 将客户端的 IP 地址字符串复制到 client_IP 数组中
		if (UserIsForbidden(client_IP))
		{
			printf("***********此IP已被禁用***************\n");
			closesocket(acceptSocket);
			exit(0);
		}


		lpProxyParam = new ProxyParam;//在堆上创建一个 ProxyParam 结构体对象
		if (lpProxyParam == NULL) {
			continue;
		}
		lpProxyParam->clientSocket = acceptSocket;// 将 acceptSocket 套接字赋值给 lpProxyParam 指向的结构体对象的 clientSocket 成员变量

		hThread = (HANDLE)_beginthreadex(NULL, 0,
			&ProxyThread, (LPVOID)lpProxyParam, 0, 0);// 创建一个新的代理线程，并返回其句柄
		CloseHandle(hThread);
		Sleep(500);
	}
	closesocket(ProxyServer);
	WSACleanup();
	return 0;
}

// Qualifier: 初始化套接字
BOOL InitSocket() {
	//加载套接字库（必须）
	WORD wVersionRequested;
	WSADATA wsaData; // 存储套接字库的信息
	//套接字加载时错误提示
	int err;
	//版本 2.2
	wVersionRequested = MAKEWORD(2, 2);
	//加载 dll 文件 Scoket 库
	err = WSAStartup(wVersionRequested, &wsaData);//调用 WSAStartup 函数来初始化套接字库
	if (err != 0) {// 如果 err 变量不等于 0，表示初始化套接字库失败
		//找不到 winsock.dll
		printf("加载 winsock 失败， 错误代码为: %d\n", WSAGetLastError());
		return FALSE;
	}
	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)// 如果 wsaData 结构体中的 wVersion 成员变量的低字节或高字节不等于 2，表示不能找到正确的套接字库版本
	{
		printf("不能找到正确的 winsock 版本\n");
		WSACleanup();
		return FALSE;
	}
	ProxyServer = socket(AF_INET, SOCK_STREAM, 0);// 调用 socket 函数来创建一个代理服务器的套接字，并将返回值赋值给 ProxyServer 变量
	if (INVALID_SOCKET == ProxyServer) {//如果 ProxyServer 变量等于 INVALID_SOCKET 值，表示创建套接字失败
		printf("创建套接字失败，错误代码为：%d\n", WSAGetLastError());
		return FALSE;
	}

	ProxyServerAddr.sin_family = AF_INET;// 于将 AF_INET 值赋值给 ProxyServerAddr 结构体中的 sin_family 成员变量，表示地址族为 IPv4
	ProxyServerAddr.sin_port = htons(ProxyPort);// 将 ProxyPort 常量转换为网络字节序后赋值给 ProxyServerAddr 结构体中的 sin_port 成员变量，表示端口号为 ProxyPort
	ProxyServerAddr.sin_addr.S_un.S_addr = INADDR_ANY;// 将 INADDR_ANY 值赋值给 ProxyServerAddr 结构体中的 sin_addr.S_un.S_addr 成员变量，表示 IP 地址为任意地址

	if (bind(ProxyServer, (SOCKADDR*)&ProxyServerAddr, sizeof(SOCKADDR)) == SOCKET_ERROR) {// 如果调用 bind 函数将代理服务器的套接字与地址信息绑定失败
		printf("绑定套接字失败\n");
		return FALSE;
	}
	if (listen(ProxyServer, SOMAXCONN) == SOCKET_ERROR) {// 如果调用 listen 函数将代理服务器的套接字设置为监听状态失败
		printf("监听端口%d 失败", ProxyPort);
		return FALSE;
	}
	return TRUE;
}

// Qualifier: 线程执行函数
unsigned int __stdcall ProxyThread(LPVOID lpParameter) {
	char Buffer[MAXSIZE], fileBuffer[MAXSIZE];//存储从客户端或服务器接收到的报文
	ZeroMemory(Buffer, MAXSIZE);
	char sendBuffer[MAXSIZE];//存储要发送给客户端或服务器的数据
	ZeroMemory(sendBuffer, MAXSIZE);
	char FishBuffer[MAXSIZE];/// 存储要发送给客户端的钓鱼网站响应数据
	ZeroMemory(FishBuffer, MAXSIZE);
	char* CacheBuffer;// 指向缓存中的数据
	SOCKADDR_IN clientAddr;// 存储客户端的地址信息
	int length = sizeof(SOCKADDR_IN);// 表示 clientAddr 结构体的大小
	int recvSize;// 存储接收到的数据大小
	int ret;// 存储发送或接收函数的返回值

	//接收客户端的请求
	recvSize = recv(((ProxyParam*)lpParameter)->clientSocket, Buffer, MAXSIZE, 0);// 调用 recv 函数从客户端套接字接收数据

	memcpy(sendBuffer, Buffer, recvSize);// 将 Buffer 数组中的 recvSize 个字节复制到 sendBuffer 数组中

	HttpHeader* httpHeader = new HttpHeader(); // 在堆上创建一个 HttpHeader 结构体对象

	CacheBuffer = new char[recvSize + 1]; //在堆上创建一个 recvSize + 1 大小的字符数组，并将其地址赋值给 CacheBuffer 指针变量
	ZeroMemory(CacheBuffer, recvSize + 1);
	memcpy(CacheBuffer, Buffer, recvSize);// 将 Buffer 数组中的 recvSize 个字节复制到 CacheBuffer 指向的数组中

	ParseHttpHead(CacheBuffer, httpHeader);//调用 ParseHttpHead 函数解析 HTTP 请求头并存储到 httpHeader 指向的结构体中


	//缓存
	char* DateBuffer;// 指向日期字段数据
	DateBuffer = (char*)malloc(MAXSIZE); // 在堆上分配 MAXSIZE 大小的内存空间，并将其地址赋值给 DateBuffer 指针变量
	ZeroMemory(DateBuffer, strlen(Buffer) + 1);
	memcpy(DateBuffer, Buffer, strlen(Buffer) + 1);// 将 Buffer 数组中的 strlen(Buffer) + 1 个字节复制到 DateBuffer 指向的数组中
	char filename[100];//存储缓存文件名
	ZeroMemory(filename, 100);
	makeFilename(httpHeader->url, filename);// 调用 makeFilename 函数根据 URL 生成缓存文件名，并存储到 filename 数组中
	char* field = (char*)"Date";// 指向日期字段的名称字符串
	char date_str[30];  //保存字段Date的值
	ZeroMemory(date_str, 30);
	ZeroMemory(fileBuffer, MAXSIZE);

	FILE* in;//读取本地缓存
	if ((in = fopen(filename, "rb")) != NULL) {
		printf("\n**********读取本地缓存****************\n");
		fread(fileBuffer, sizeof(char), MAXSIZE, in);
		fclose(in);
		ParseDate(fileBuffer, field, date_str);//读取缓存文件中的 HTTP 响应头，并提取其中的日期字段
		printf("date_str:%s\n", date_str);
		makeNewHTTP(Buffer, date_str);
		haveCache = TRUE;
		goto success;
	}
	delete CacheBuffer;

	if (!ConnectToServer(&((ProxyParam*)lpParameter)->serverSocket, httpHeader->host)) {// 调用 ConnectToServer 函数连接到目标服务器失败
		goto error;
	}
	printf("代理连接主机 %s 成功\n", httpHeader->host);


	//屏蔽网站信息
	if (ForbiddenToConnect(httpHeader->url))// 调用 ForbiddenToConnect 函数判断是否禁止访问某个网站
	{
		printf("*************不允许访问 %s *******************\n", httpHeader->url);
		goto error;

	}

	//网站引导  访问http://pku.edu.cn/  重定向到 http://today.hit.edu.cn/
	if (GotoFalseWebsite(httpHeader->url))
	{

		char* pr;// 指向 FishBuffer 数组中的当前位置
		int fishing_len = 0;//使用fishing_len来记录已读取报文的长度，以方便接下来修改后面报文
		fishing_len = strlen("HTTP/1.1 302 Moved Temporarily\r\n");
		memcpy(FishBuffer, "HTTP/1.1 302 Moved Temporarily\r\n", fishing_len);
		pr = FishBuffer + fishing_len;
		fishing_len = strlen("Connection:keep-alive\r\n");
		memcpy(pr, "Connection:keep-alive\r\n", fishing_len);
		pr = pr + fishing_len;// 将 pr 变量加上 fishing_len 的值，表示移动到下一个位置
		fishing_len = strlen("Cache-Control:max-age=0\r\n");
		memcpy(pr, "Cache-Control:max-age=0\r\n", fishing_len);
		pr = pr + fishing_len;
		//重定向到今日哈工大
		fishing_len = strlen("Location: http://today.hit.edu.cn/\r\n\r\n");
		memcpy(pr, "Location: http://today.hit.edu.cn/\r\n\r\n", fishing_len);
		//将302报文返回给客户端
		ret = send(((ProxyParam*)lpParameter)->clientSocket, FishBuffer, sizeof(FishBuffer), 0); // 调用 send 函数向客户端套接字发送钓鱼网站响应数据
		goto error;
	}
	if (recvSize <= 0) {
		goto error;
	}

success://有缓存直接读取后发送给客户端
	if (!ConnectToServer(&((ProxyParam*)lpParameter)->serverSocket, httpHeader->host)) {// HttpHeader 结构体中的主机名创建并连接到原服务器套接字
		printf("连接目标服务器失败！！！\n");
		goto error;
	}
	printf("代理连接主机 %s 成功\n", httpHeader->host);
	//将客户端发送的 HTTP 数据报文直接转发给目标服务器
	ret = send(((ProxyParam*)lpParameter)->serverSocket, Buffer, strlen(Buffer) + 1, 0);
	//等待目标服务器返回数据
	recvSize = recv(((ProxyParam*)lpParameter)->serverSocket, Buffer, MAXSIZE, 0);
	if (recvSize <= 0) {
		printf("返回目标服务器的数据失败！！！\n");
		goto error;
	}
	//有缓存时，判断返回的状态码是否是304，若是则将缓存的内容发送给客户端
	if (haveCache == TRUE) {// 如果 haveCache 变量等于 TRUE，表示有缓存文件可用
		getCache(Buffer, filename);
	}
	if (needCache == TRUE) {// 如果 needCache 变量等于 TRUE，表示需要缓存文件
		makeCache(Buffer, httpHeader->url);  //缓存报文
	}
	//将目标服务器返回的数据直接转发给客户端
	ret = send(((ProxyParam*)lpParameter)->clientSocket, Buffer, sizeof(Buffer), 0);


	//错误处理
error:
	printf("关闭套接字\n");
	Sleep(200);
	closesocket(((ProxyParam*)lpParameter)->clientSocket);
	closesocket(((ProxyParam*)lpParameter)->serverSocket);
	delete lpParameter;
	_endthreadex(0);
	return 0;

}

//Qualifier:实现网站过滤，不允许访问某些网站
bool ForbiddenToConnect(char* httpheader)
{
	char* forbiddernUrl = (char*)"http://www.hit.edu.cn/";
	if (!strcmp(httpheader, forbiddernUrl))
	{
		return true;
	}
	else
		return false;
}


//Qualifier:实现用户过滤，禁用IP
bool UserIsForbidden(char* userID)
{
	for (int i = 0; i < IPnum; i++)// 遍历被禁止访问代理服务器的 IP 地址数组
	{
		if (strcmp(userID, ForbiddenIP[i]) == 0)
		{
			//用户IP在禁用IP表中
			return true;
		}
	}
	return false;
}

//Qualifier:实现访问引导到模拟网站
bool GotoFalseWebsite(char* url)
{
	cout << url << endl;
	for (int i = 0; i < fishUrlnum; i++)// 从 0 到 fishUrlnum - 1 遍历钓鱼网站 URL 数组
	{
		if (strcmp(url, fishUrl[i]) == 0)
		{
			return true;
		}
	}
	return false;
}


// Qualifier: 解析 TCP 报文中的 HTTP 头部
void ParseHttpHead(char* buffer, HttpHeader* httpHeader) {
	char* p;//指向分割后的字符串
	char* ptr;//存储分割函数的上下文信息
	const char* delim = "\r\n";

	p = strtok_s(buffer, delim, &ptr);//调用 strtok_s 函数从 buffer 指向的字符串中分割出第一个子字符串

	if (p[0] == 'G') {//GET 方式
		memcpy(httpHeader->method, "GET", 3);
		memcpy(httpHeader->url, &p[4], strlen(p) - 13);
	}
	else if (p[0] == 'P') {//POST 方式
		memcpy(httpHeader->method, "POST", 4);
		memcpy(httpHeader->url, &p[5], strlen(p) - 14);
	}

	p = strtok_s(NULL, delim, &ptr);//调用 strtok_s 函数从 buffer 指向的字符串中继续分割出下一个子字符串

	while (p) {
		switch (p[0]) {
		case 'H':// 表示 HTTP 请求头中的主机名字段
			memcpy(httpHeader->host, &p[6], strlen(p) - 6);
			break;
		case 'C':// 表示 HTTP 请求头中可能有 Cookie 字段
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


// Qualifier: 根据主机创建目标服务器套接字，并连接
BOOL ConnectToServer(SOCKET* serverSocket, char* host) {
	sockaddr_in serverAddr;//存储目标服务器的地址信息
	serverAddr.sin_family = AF_INET;// IPv4 协议
	serverAddr.sin_port = htons(HTTP_PORT);//使用 HTTP 端口号
	HOSTENT* hostent = gethostbyname(host);//主机名字符串
	if (!hostent) {
		return FALSE;
	}
	in_addr Inaddr = *((in_addr*)*hostent->h_addr_list);//获取目标服务器的 IP 地址
	serverAddr.sin_addr.s_addr = inet_addr(inet_ntoa(Inaddr));//设置目标服务器的 IP 地址
	*serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (*serverSocket == INVALID_SOCKET) {
		return FALSE;
	}
	if (connect(*serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr))
		== SOCKET_ERROR) {
		closesocket(*serverSocket);
		return FALSE;
	}
	return TRUE;
}


//分析HTTP头部的field字段，如果包含该field则返回true，并获取日期
BOOL ParseDate(char* buffer, char* field, char* tempDate) {
	char* p, * ptr, temp[5];// 指向分割后的字符串, 存储分割函数的上下文信息, 临时数据
	const char* delim = "\r\n";
	ZeroMemory(temp, 5);
	p = strtok(buffer, delim);
	int len = strlen(field) + 2;
	while (p) {
		if (strstr(p, field) != NULL) {
			memcpy(tempDate, &p[len], strlen(p) - len);
			return TRUE;
		}
		p = strtok(NULL, delim);
	}
	return TRUE;
}

//改造HTTP请求报文
void makeNewHTTP(char* buffer, char* value) {
	const char* field = "Host";// HTTP 请求头中主机名
	const char* newfield = "If-Modified-Since: ";//HTTP 请求头中日期字段
	char temp[MAXSIZE];
	ZeroMemory(temp, MAXSIZE);
	char* pos = strstr(buffer, field);//在 buffer 指向的字符串中查找 field 指向的字符串
	int i = 0;
	for (i = 0; i < strlen(pos); i++) {
		temp[i] = pos[i];
	}
	*pos = '\0';
	while (*newfield != '\0') {  //插入If-Modified-Since字段
		*pos++ = *newfield++;
	}
	while (*value != '\0') {
		*pos++ = *value++;
	}
	*pos++ = '\r';
	*pos++ = '\n';
	for (i = 0; i < strlen(temp); i++) {
		*pos++ = temp[i];
	}
}

//根据url构造文件名
void makeFilename(char* url, char* filename) {
	while (*url != '\0') {
		if (*url != '/' && *url != ':' && *url != '.') {
			*filename++ = *url;//将 url 指向的字符复制到 filename 指向的位置
		}
		url++;
	}
	strcat(filename, ".txt");
}

//进行缓存
void makeCache(char* buffer, char* url) {
	char* p, * ptr, num[10], tempBuffer[MAXSIZE + 1]; // 指向分割后的字符串, 存储分割函数的上下文信息, 状态码字符串, 临时数据
	const char* delim = "\r\n";
	ZeroMemory(num, 10);
	ZeroMemory(tempBuffer, MAXSIZE + 1);
	memcpy(tempBuffer, buffer, strlen(buffer));
	p = strtok(tempBuffer, delim);//分割出第一个子字符串
	memcpy(num, &p[9], 3);
	if (strcmp(num, "200") == 0) {  //状态码是200时缓存
		char filename[1024] = { 0 };
		makeFilename(url, filename);// 根据 url 指向的字符串生成缓存文件名
		printf("filename : %s\n", filename);
		ofstream of;
		of.open(filename, ios::out);
		of << buffer << endl;//将 buffer 指向的字符串写入到 of 对象中
		of.close();
		printf("\n=====================================\n\n");
		printf("\n***********网页已经被缓存**********\n");
	}
}

//获取缓存
void getCache(char* buffer, char* filename) {
	char* p, * ptr, num[10], tempBuffer[MAXSIZE + 1];//指向分割后的字符串,存储分割函数的上下文信息,状态码字符串,临时数据
	const char* delim = "\r\n";
	ZeroMemory(num, 10);
	ZeroMemory(tempBuffer, MAXSIZE + 1);
	memcpy(tempBuffer, buffer, strlen(buffer));
	p = strtok(tempBuffer, delim);//分割出第一个子字符串
	memcpy(num, &p[9], 3);
	if (strcmp(num, "304") == 0) {  //主机返回的报文中的状态码为304时返回已缓存的内容
		printf("\n=====================================\n\n");
		printf("***********从本机获得缓存**************\n");
		ZeroMemory(buffer, strlen(buffer));
		FILE* in = NULL;
		if ((in = fopen(filename, "r")) != NULL) {
			fread(buffer, sizeof(char), MAXSIZE, in);
			fclose(in);
		}
		needCache = FALSE;
	}
}