#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS


// GBN_client.cpp : 定义控制台应用程序的入口点。
//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <WinSock2.h>
#include <time.h>
#include <fstream>
#include <string>


#pragma comment(lib,"ws2_32.lib")

#define SERVER_PORT 12340 //接收数据的端口号
#define SERVER_PORT_S 12315 //接收数据的端口号
#define SERVER_IP  "127.0.0.1" // 服务器的IP 地址
const int BUFFER_LENGTH = 1026;
const int SEQ_SIZE = 20;//接收端序列号个数， 为 1~20
//-------------------------------------------
const int SEND_WIND_SIZE = 10;//发送窗口大小为 10 ， GBN 中应满足 W + 1 <=N（W 为发送窗口大小， N 为序列号个数）
BOOL ack[SEQ_SIZE];//收到ack 情况， 对应0~19 的ack
int curSeq;//当前数据包的 seq
int  curAck;//当前等待确认的 ack
int totalSeq;//收到的包的总数
int totalPacket;//需要发送的包总数

/****************************************************************/
/* -time 从服务器端获取当前时间
-quit 退出客户端
-gbnsc [X] [Y]测试GBN (Server2Client) 协议实现可靠数据传输
-gbncs [X] [Y]测试GBN (Client2Server) 协议实现可靠数据传输
[X] [0,1] 模拟数据包丢失的概率
[Y] [0,1] 模拟ACK 丢失的概率
*/
/****************************************************************/
void printTips() {
	printf("*****************************************\n");
	printf("| -time to get current time| \n");
	printf("| -quit to exit client |\n");
	printf("| -gbnsc [X] [Y] to test the gbn |\n");
	printf("| -gbncs [X] [Y] to test the gbn |\n");
	printf("*****************************************\n");
}

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

//客户端发送时需要的函数
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
		for (int i = 0; i <= index; ++i) {
			ack[i] = TRUE;
		}
		curAck = index + 1;
	}
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
	if (err != 0) {
		//找不到winsock.dll
		printf("WSAStartup failed with error: %d\n", err);
		return 1;
	}
	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
	{
		printf("Could not find a usable version of Winsock.dll\n");
		WSACleanup();
	}
	else {
		printf("The Winsock 2.2 dll was found okay\n");
	}

	SOCKET socketClient = socket(AF_INET, SOCK_DGRAM, 0);
	int iMode = 0; //1： 非阻塞， 0： 阻塞
	ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & iMode);//非阻塞设置，函数调用将立即返回，而不会等待操作完成

	SOCKADDR_IN addrServer;//服务器网络地址
	addrServer.sin_addr.S_un.S_addr =inet_addr(SERVER_IP);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(SERVER_PORT);
	
	//接收或发送缓冲区
	char buffer[BUFFER_LENGTH];
	ZeroMemory(buffer, sizeof(buffer));
	int len = sizeof(SOCKADDR);

	//为了测试与服务器的连接， 可以使用 -time 命令从服务器端获得当前时间
	//使用 -testgbn [X] [Y] 测试GBN 其中[X]表示数据包丢失概率 [Y]表示ACK 丢包概率
	printTips();
	int ret;
	int interval = 1;//收到数据包之后返回ack 的间隔， 默认为 1 表示每个都返回ack ， 0 或者负数均表示所有的都不返回ack
	char cmd[128];//接收输入的指令
	float packetLossRatio = 0.2; //默认包丢失率0.2
	float ackLossRatio = 0.2; //默认ACK 丢失率0.2

	//用时间作为随机种子， 放在循环的最外面
	srand((unsigned)time(NULL));

	BOOL ISDOWN = true;//判断是否传输完成

	//客户端作为发送方时，将test.txt中的数据读入内存
	std::ifstream  send;
	send.open("./testc.txt", std::ios::out);
	char data[1024 * 113];
	ZeroMemory(data, sizeof(data));
	send.read(data, 1024 * 113);
	send.close();
	//printf("测试:%s\n", data);
	totalPacket = sizeof(data) / 1024;
	int recvSize;

	//将所有ack设为true
	for (int i = 0; i < SEQ_SIZE; ++i) {
		ack[i] = TRUE;
	}

	while (true) {
		gets_s(buffer, BUFFER_LENGTH);
		ret = sscanf(buffer, "%s%f%f", &cmd, &packetLossRatio, &ackLossRatio);//输入指令
		//开始GBN 测试， 使用GBN 协议实现UDP 可靠文件传输
		if (!strcmp(cmd, "-gbnsc")) {
			printf("%s\n", "Begin to test GBN S_To_C protocol, please don't abort the process");
			printf("The loss ratio of packet is %.2f,the loss ratio of ack is % .2f\n", packetLossRatio, ackLossRatio);
			int waitCount = 0;
			int stage = 0;//客户端阶段
			BOOL IS_LOST;
			unsigned char u_code;//状态码
			unsigned short seq;//包的序列号
			unsigned short recvSeq;//接收窗口大小为 1， 已确认的序列号
			unsigned short waitSeq;//等待的序列号
			sendto(socketClient, "-gbnsc", strlen("-gbnsc") + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));

			std::ofstream recv;	//ofstream以写入模式打开文件
			recv.open("./recvC.txt", std::ios::out | std::ios::trunc);//out写入,trunc覆写
			if (!recv.is_open()) {
				printf("文件打开失败！\n");
				return 1;
			}
	
			while (true)
			{ //等待 server 回复设置UDP 为阻塞模式
				recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, &len);
				switch (stage) {
				case 0://等待握手阶段
					u_code = (unsigned char)buffer[0];
					if ((unsigned char)buffer[0] == 205)
					{
						printf("Ready for file transmission\n");
						buffer[0] = 200;
						buffer[1] = '\0';
						sendto(socketClient, buffer, 2, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
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
						printf("C loss packet %d\n", seq);//服务器传向客户端的包丢失
						continue;
					}

					//如果是期待的包， 正确接收， 正常确认即可
					printf("C recv packet %d\n", seq);

					//如果收到的包序号与等待的一致
					if (!(waitSeq - seq)) { 
						//客户端接收数据完毕
						if (!(strlen(buffer) - 1)&& ISDOWN)
						{
							printf("%s\n", "接收完毕");
							ISDOWN = false;
						}
						//等收到的包序号与等待的一致，再写入文件，就不会乱序输入了
						for (int i = 1; i < strlen(buffer)-1; i++)//用strlen(buffer)-1限制无效空字符的输入
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
					sendto(socketClient, buffer, 2, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
					printf("C send ack    %d\n", (unsigned char)buffer[0]);
					break;
				}Sleep(500);
			}recv.close();//关闭文件
		}
		if (!strcmp(cmd, "-gbncs")) {

			int new_iMode = 1; //1： 非阻塞， 0： 阻塞
			ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & new_iMode);//非阻塞设置，函数调用将立即返回，而不会等待操作完成
			//进入gbn 测试阶段
			//首先 server（server 处于0 状态） 向client 发送205 状态码（server进入 1 状态）
			//server 等待client 回复200 状态码， 如果收到（server 进入2 状态） ，则开始传输文件， 否则延时等待直至超时\
			//在文件传输阶段， server 发送窗口大小设为
			ZeroMemory(buffer, sizeof(buffer));
			int recvSize;
			int waitCount = 0;//计时器
			printf("Begainto test GBN C_To_S protocol,please don't abort the process\n");
			//加入了一个握手阶段
			//首先服务器向客户端发送一个 205 大小的状态码（我自己定义的）表示服务器准备好了，可以发送数据
			//客户端收到 205 之后回复一个 200 大小的状态码， 表示客户端准 备好了，可以接收数据了
			//服务器收到200 状态码之后， 就开始使用GBN 发送数据了
			printf("Shake hands stage\n");
			sendto(socketClient, "-gbncs", strlen("-gbncs") + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
			int stage = 0;
			bool runFlag = true;
			while (runFlag) {
				switch (stage) {
				case 0://发送205 阶段 
					buffer[0] = 205;
					sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
					Sleep(100);
					stage = 1;
					break;
				case 1://等待接收200 阶段， 没有收到则计数器+1， 超时则放弃此次“连接” ， 等待从第一步开始
					recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
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
							printf("File size is %dB, each packet is 1024B and packet total num is % d\n", sizeof(data), totalPacket);
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
						printf("C send packet %d\n", curSeq);
						//printf("%s\n", &buffer[1]);
						sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));//将包传给客户端
						++curSeq; //将当前序列号 curSeq 增加1
						curSeq %= SEQ_SIZE; //将其对总序列号个数取模，以循环使用序列号
						++totalSeq;
						Sleep(500);
					}
					//等待Ack， 若没有收到， 则返回值为 - 1， 计数器 + 1
					recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
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
		//输入time时
		sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
		ret = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, &len);
		//输入quit时
		printf("%s\n", buffer);
		if (!strcmp(buffer, "Goodbye!")) {
			break;
		}
		//其他输入
		printTips();
	}
	//关闭套接字
	closesocket(socketClient);
	WSACleanup();
	return 0;
}