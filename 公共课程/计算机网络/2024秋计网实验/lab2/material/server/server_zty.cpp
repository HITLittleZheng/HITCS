#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS

//#include "stdafx.h" //创建VS 项目包含的预编译头文件
#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <fstream>
#include <string>
#pragma comment(lib,"ws2_32.lib")

#define SERVER_PORT 12340 //端口号
#define SERVER_IP "0.0.0.0" //IP 地址
const int BUFFER_LENGTH = 1026; //缓冲区大小，（以太网中 UDP 的数据帧中包长度应小于 1480 字节）
const int SEND_WIND_SIZE = 10;//发送窗口大小为 10 ， GBN 中应满足 W + 1 <=N（W 为发送窗口大小， N 为序列号个数）
//本例取序列号 0...19 共20 个
//如果将窗口大小设为 1， 则为停-等协议
const int SEQ_SIZE = 20; //序列号的个数， 从0~19 共计20 个
//由于发送数据第一个字节如果值为0， 则数据会发送失败
//因此接收端序列号为 1~20， 与发送端一一对应
BOOL ack[SEQ_SIZE];//收到ack 情况， 对应0~19 的ack
int curSeq;//当前数据包的 seq
int  curAck;//当前等待确认的 ack
int totalSeq;//收到的包的总数
int totalPacket;//需要发送的包总数

//************************************
// Method: getCurTime
// FullName: getCurTime
// Access: public
// Returns: void
// Qualifier: 获取当前系统时间， 结果存入ptime 中
// Parameter: char * ptime
//************************************
void getCurTime(char* ptime) {
	char buffer[128];
	memset(buffer, 0, sizeof(buffer));
	time_t c_time;
	struct tm* p;
	time(&c_time);
	p = localtime(&c_time);
	sprintf_s(buffer, "%d/%d/%d %d:%d:%d",
		p->tm_year + 1900,
		p->tm_mon,
		p->tm_mday,
		p->tm_hour,
		p->tm_min,
		p->tm_sec);
	strcpy_s(ptime, sizeof(buffer), buffer);
} 

//************************************
// Method: seqIsAvailable
// FullName: seqIsAvailable
// Access: public
// Returns: bool
// Qualifier: 当前序列号 curSeq 是否可用
//************************************
bool seqIsAvailable() {
	int step;
	//计算当前序列号curSeq和等待确认的ACK序列号 curAck 之间的步数差
	step = curSeq - curAck;

	step = step >= 0 ? step : step + SEQ_SIZE;
	//序列号是否在当前发送窗口之内
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
// FullName: timeoutHandler
// Access: public
// Returns: void
// Qualifier: 超时重传处理函数， 滑动窗口内的数据帧都要重传
//************************************
void timeoutHandler() {
	printf("Timer out error.\n");
	int index;
	/*
		通过一个循环，将从当前等待确认的ACK序列号 curAck 开始的连续 SEND_WIND_SIZE 个位置的ACK标记设置为TRUE，
		表示这些数据包需要被重传。
	*/
	for (int i = 0; i < SEND_WIND_SIZE; ++i) {
		index = (i + curAck) % SEQ_SIZE;
		ack[index] = TRUE;
	} 
	//更新收到的数据包总数totalSeq，减去因为超时而需要重传的数据包数量
	totalSeq -= SEND_WIND_SIZE;
	//当前数据包序列号回滚到窗口的起始位置，以便开始下一轮的发送
	curSeq = curAck;
} 

//************************************
// Method: ackHandler
// FullName: ackHandler
// Access: public
// Returns: void
// Qualifier: 收到ack， 累积确认， 取数据帧的第一个字节
//由于发送数据时， 第一个字节（序列号） 为 0（ASCII） 时发送失败， 因此加一了， 此处需要减一还原
// Parameter: char c
//************************************
void ackHandler(char c) {
	unsigned char index = (unsigned char)c - 1; //序列号减一
	printf("Recv a ack of %d\n", index);
	if (curAck <= index) {
		for (int i = curAck; i <= index; ++i) {
			ack[i] = TRUE;
		} 
		curAck = (index + 1) % SEQ_SIZE;
	}
	else {
		//ack 超过了最大值， 回到了curAck 的左边
		for (int i = curAck; i < SEQ_SIZE; ++i) {
			ack[i] = TRUE;
		} 
		for(int i = 0; i <= index; ++i) {
			ack[i] = TRUE;
		} 
		curAck = index + 1;
	}
} 

//作为接收方时用的函数
//************************************
// Method:lossInLossRatio
// FullName:lossInLossRatio
// Access:public
// Returns:BOOL
// Qualifier: 根据丢失率随机生成一个数字， 判断是否丢失, 丢失则返回 TRUE， 否则返回FALSE
// Parameter: float lossRatio [0,1]
//************************************
BOOL lossInLossRatio(float lossRatio) {
	int lossBound = (int)(lossRatio * 100);
	int r = rand() % 101;
	if (r < lossBound) {
		return TRUE;
	}
	return FALSE;
}

//主函数
int main(int argc, char* argv[])
{
	//加载套接字库（必须）
	WORD wVersionRequested;
	WSADATA wsaData;
	//套接字加载时错误提示
	int err;
	//版本2.2
	wVersionRequested = MAKEWORD(2, 2);
	//加载dll 文件Scoket 库
	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) 
	{
		//找不到winsock.dll
		printf("WSAStartup failed with error: %d\n", err);
		return -1;
	} 
	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
	{
		printf("Could not find a usable version of Winsock.dll\n");
		WSACleanup();
	}
	else {
		printf("The Winsock 2.2 dll was found okay\n");
	} 

	//创建一个 UDP（用户数据报协议）类型的套接字，用于在 IPv4 网络上进行通信
	SOCKET sockServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	//设置套接字为非阻塞模式
	int iMode = 1; //1： 非阻塞， 0： 阻塞
	ioctlsocket(sockServer, FIONBIO, (u_long FAR*) & iMode);//非阻塞设置，函数调用将立即返回，而不会等待操作完成

	SOCKADDR_IN addrServer; //服务器地址
	//addrServer.sin_addr.S_un.S_addr = inet_addr(SERVER_IP);
	addrServer.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//两者均可，INADDR_ANY == 0.0.0.0，表示本机的所有IP
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(SERVER_PORT);	 //将一个无符号短整型数值转换为TCP/IP网络字节序，即大端模式(big-endian)

	//绑定套接字与服务器地址
	err = bind(sockServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	if (err) {
		err = GetLastError();
		printf("Could not bind the port %d for socket.Error code is % d\n",SERVER_PORT,err);
		WSACleanup();
		return -1;
	} 

	SOCKADDR_IN addrClient; //客户端地址
	int length = sizeof(SOCKADDR);
	char buffer[BUFFER_LENGTH]; //数据发送接收缓冲区
	ZeroMemory(buffer, sizeof(buffer));

	float packetLossRatio = 0.2; //默认包丢失率0.2
	float ackLossRatio = 0.2; //默认ACK 丢失率0.2
	//用时间作为随机种子， 放在循环的最外面
	srand((unsigned)time(NULL));

	//将test.txt中的数据读入内存
	std::ifstream send;
	send.open("./tests.txt", std::ios::out);
	char data[1024 * 113];
	ZeroMemory(data, sizeof(data));
	send.read(data, 1024 * 113);
	send.close();
	//printf("测试:%s\n", data);
	totalPacket = sizeof(data) / 1024;
	int recvSize;

	BOOL ISDOWN = true;//判断是否传输完成

	//将所有ack设为true
	for (int i = 0; i < SEQ_SIZE; ++i) {
		ack[i] = TRUE;
	} 

	while(true) {
		//非阻塞接收， 若没有收到数据， 返回值为-1
		// recvfrom函数从套接字 sockServer 接收数据，并将接收到的数据存储到缓冲区 buffer 中，同时获取发送方的地址信息，并将其存储到 addrClient 中。
		recvSize = recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrClient), &length);
		printf("%s", buffer);
		if (recvSize < 0) {
			Sleep(200);
			continue;
		} 
		printf("recv from client: %s\n", buffer);
		if (strcmp(buffer, "-time") == 0) {
			getCurTime(buffer);
		}
		else if (strcmp(buffer, "-quit") == 0) {
			strcpy_s(buffer, strlen("Goodbye!") + 1, "Goodbye!");
		}
		else if (strcmp(buffer, "-gbnsc") == 0) {
			//进入gbn 测试阶段
			//首先 server（server 处于0 状态） 向client 发送205 状态码（server进入 1 状态）
			//server 等待client 回复200 状态码， 如果收到（server 进入2 状态） ，则开始传输文件， 否则延时等待直至超时\
			//在文件传输阶段， server 发送窗口大小设为
			ZeroMemory(buffer, sizeof(buffer));
			int recvSize;
			int waitCount = 0;//计时器
			printf("Begainto test GBN S_To_C protocol,please don't abort the process\n");
			//加入了一个握手阶段
			//首先服务器向客户端发送一个 205 大小的状态码（我自己定义的）表示服务器准备好了，可以发送数据
			//客户端收到 205 之后回复一个 200 大小的状态码， 表示客户端准 备好了，可以接收数据了
			//服务器收到200 状态码之后， 就开始使用GBN 发送数据了
			printf("Shake hands stage\n");
			int stage = 0;
			bool runFlag = true;
			while (runFlag) {
				switch (stage) {
				case 0://发送205 阶段 
					buffer[0] = 205;
					sendto(sockServer,	buffer, strlen(buffer) + 1,0,(SOCKADDR*)&addrClient, sizeof(SOCKADDR));
					Sleep(100);
					stage = 1;
					break;
				case 1://等待接收200 阶段， 没有收到则计数器+1， 超时则放弃此次“连接” ， 等待从第一步开始
					recvSize =recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrClient), &length);
					if (recvSize < 0) {
						++waitCount;
						if (waitCount > 20) {
							runFlag = false;
							printf("Timeout error\n");
							break;
						} 
						Sleep(500);
						continue;
					}
					else {
						if ((unsigned char)buffer[0] == 200) {
							printf("Begin a file transfer\n");
							printf("File size is %dB, each packet is 1024B and packet total num is % d\n",sizeof(data),totalPacket);
							curSeq = 0;
							curAck = 0;
							totalSeq = 0;
							waitCount = 0;
							stage = 2;
						}
					} 
					break;
				//数据传输阶段
				case 2:
					/*
						seqIsAvailable()函数用于检查当前序列号 curSeq 是否可用。如果发送窗口内有足够的空间，
						并且当前序列号对应的数据包未发送过或者已经收到了确认，那么就可以发送当前序列号对应的数据包
					*/
					if (seqIsAvailable()) {	
						//发送给客户端的序列号从 1 开始
						buffer[0] = curSeq + 1;
						ack[curSeq] = FALSE;
						//数据发送的过程中应该判断是否传输完成,为简化过程此处并未实现
						memcpy(&buffer[1], data + 1024 * totalSeq, 1024);//将数据缓冲区data中的一部分数据拷贝到UDP数据包的缓冲区buffer中
						printf("S send packet %d\n", curSeq);
						sendto(sockServer, buffer, BUFFER_LENGTH,0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));//将包传给客户端
						++curSeq; //将当前序列号 curSeq 增加1
						curSeq %= SEQ_SIZE; //将其对总序列号个数取模，以循环使用序列号
						++totalSeq;
						Sleep(500);
					} 
					//等待Ack， 若没有收到， 则返回值为 - 1， 计数器 + 1
					recvSize = recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrClient), &length);
					if (recvSize < 0) {
						waitCount++;
						//20 次等待ack 则超时重传
						if (waitCount > 20)
						{
							timeoutHandler();
							waitCount = 0;
						}
					}
					else {
						//收到ack
						ackHandler(buffer[0]);
						waitCount = 0;
					}
					Sleep(500);
					break;
				}
			}
		} 
		else if (strcmp(buffer, "-gbncs") == 0) {
			printf("%s\n", "Begin to test GBN C_To_S protocol, please don't abort the process");
			printf("The loss ratio of packet is %.2f,the loss ratio of ack is % .2f\n", packetLossRatio, ackLossRatio);
			int waitCount = 0;
			int stage = 0;//客户端阶段
			BOOL IS_LOST;
			unsigned char u_code;//状态码
			unsigned short seq;//包的序列号
			unsigned short recvSeq;//接收窗口大小为 1， 已确认的序列号
			unsigned short waitSeq;//等待的序列号
			int new_iMode = 0;

			std::ofstream recv;	//ofstream以写入模式打开文件
			recv.open("./recvS.txt", std::ios::out | std::ios::trunc);//out写入,trunc覆写
			if (!recv.is_open()) {
				printf("文件打开失败！\n");
				return 1;
			}
			ioctlsocket(sockServer, FIONBIO, (u_long FAR*)& new_iMode);//阻塞设置，函数调用会等待操作完成后进行
			while (true)
			{ //等待 client 回复设置UDP 为阻塞模式
				recvfrom(sockServer, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrClient, &length);
				switch (stage) {
				case 0://等待握手阶段
					u_code = (unsigned char)buffer[0];
					if ((unsigned char)buffer[0] == 205)
					{
						printf("Ready for file transmission\n");
						buffer[0] = 200;
						buffer[1] = '\0';
						sendto(sockServer, buffer, 2, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
						stage = 1;
						recvSeq = 0;
						waitSeq = 1;//等待接收的包序号
					}
					break;
					//等待接收数据阶段
				case 1:
					seq = (unsigned short)buffer[0]; //获取收到的包序号
					//随机法模拟包是否丢失
					IS_LOST = lossInLossRatio(packetLossRatio);
					if (IS_LOST) {
						printf("S loss packet %d\n", seq);//服务器传向客户端的包丢失
						continue;
					}

					//如果是期待的包， 正确接收， 正常确认即可
					printf("S recv packet %d\n", seq);

					//如果收到的包序号与等待的一致
					if (!(waitSeq - seq)) {
						//客户端接收数据完毕
						if (!(strlen(buffer) - 1) && ISDOWN)
						{
							printf("%s\n", "接收完毕");
							ISDOWN = false;
						}
						//等收到的包序号与等待的一致，再写入文件，就不会乱序输入了
						for (int i = 1; i < strlen(buffer) - 1; i++)//用strlen(buffer)-1限制无效空字符的输入
						{
							recv << buffer[i];
						}
						++waitSeq;	//等待序列号加1
						if (waitSeq == 21) {
							waitSeq = 1;
						}
						//测试buffer长度
						//printf("%d\n", strlen(buffer));
						//if(!strlen(buffer)-1)
						//输出数据
						//printf("%s\n",&buffer[1]);
						buffer[0] = seq;
						recvSeq = seq;
						buffer[1] = '\0';
					}
					else {
						//如果当前一个包都没有收到， 则等待 Seq 为 1 的数据包， 不是则不返回ACK（因为并没有上一个正确的ACK）
						if (!recvSeq) {
							continue;
						}
						buffer[0] = recvSeq;
						buffer[1] = '\0';
					}

					//随机法模拟ack丢失
					IS_LOST = lossInLossRatio(ackLossRatio);
					if (IS_LOST) {
						printf("C ack %d loss\n", (unsigned char)buffer[0]);//客户端传向服务端的ack丢失
						continue;
					}
					//发送ack
					sendto(sockServer, buffer, 2, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
					printf("C send ack    %d\n", (unsigned char)buffer[0]);
					break;
				}Sleep(500);
			}recv.close();//关闭文件
		}
		sendto(sockServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient,sizeof(SOCKADDR));
		Sleep(500);
		}
	//关闭套接字， 卸载库
	closesocket(sockServer);
	WSACleanup();
	return 0;
}