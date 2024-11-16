#include <stdlib.h>
#include <WinSock2.h>
#include <time.h>
#include <stdio.h>
#include<fstream>
#include<sstream>
#include<cstdio>
#pragma comment(lib,"ws2_32.lib")
#pragma warning(disable:4996) 
#define SERVER_PORT 12340 //接收数据的端口号
#define SERVER_IP "127.0.0.1" // 服务器的 IP 地址
using namespace std;
const int BUFFER_LENGTH = 1027;
const int SEQ_SIZE = 20;//接收端序列号个数，为 1~20
BOOL ack[SEQ_SIZE];//收到 ack 情况，对应 0~19 的 ack
int curSeq;//当前数据包的 seq
int curAck;//当前等待确认的 ack
int totalSeq;//收到的包的总数
int totalPacket;//需要发送的包总数
int waitSeq;
const int SEND_WIND_SIZE = 10;
/****************************************************************/
/* -time 从服务器端获取当前时间
-quit 退出客户端
-testgbn [X] 测试 GBN 协议实现可靠数据传输
[X] [0,1] 模拟数据包丢失的概率
[Y] [0,1] 模拟 ACK 丢失的概率
*/
/****************************************************************/
void printTips() {
	printf("| -time to get current time |\n");
	printf("| -quit to exit client |\n");
	printf("| gbn + [X] +[Y] + op  +filename |\n");
	printf("| sr + [X] +[Y] + op  +filename  |\n");
	printf("*****************************************\n");
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
	unsigned char index = (unsigned char)c - 1; //序列号减一
	printf("Recv a ack of %d\n", index);
	if (curAck <= index) {
		for (int i = curAck; i <= index; ++i) {
			ack[i] = TRUE;
		}
		curAck = (index + 1) % SEQ_SIZE;
	}
	else {
		//ack 超过了最大值，回到了 curAck 的左边
		for (int i = curAck; i < SEQ_SIZE; ++i) {
			ack[i] = TRUE;
		}
		for (int i = 0; i <= index; ++i) {
			ack[i] = TRUE;
		}
		curAck = index + 1;
	}
}
int main()
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
		printf("WSAStartup failed with error: %d\n", err);
		return 1;
	}
	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
		printf("Could not find a usable version of Winsock.dll\n");
		WSACleanup();
	}
	else {
		printf("The Winsock 2.2 dll was found okay\n");
	}
	SOCKET socketClient = socket(AF_INET, SOCK_DGRAM, 0);
	SOCKADDR_IN addrServer;
	addrServer.sin_addr.S_un.S_addr = inet_addr(SERVER_IP);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(SERVER_PORT);
	//接收缓冲区
	char buffer[BUFFER_LENGTH];
	ZeroMemory(buffer, sizeof(buffer));
	int len = sizeof(SOCKADDR);
	//为了测试与服务器的连接，可以使用 -time 命令从服务器端获得当前时间
		//使用 -testgbn [X] [Y] 测试 GBN 其中[X]表示数据包丢失概率
		// [Y]表示 ACK 丢包概率
	int ret;
	int interval = 1;//收到数据包之后返回 ack 的间隔，默认为 1 表示每个都返回 ack，0 或者负数均表示所有的都不返回 ack
	char cmd[128];
	float packetLossRatio = 0.1; //默认包丢失率 0.2
	float ackLossRatio = 0.1; //默认 ACK 丢失率 0.2
	char operation[10];
	char filename[100];
	int sendack = 0;
	int iMode = 0;
	int loct = 0;
	int waitCount = 0;

	srand((unsigned)time(NULL));
	while (true) {
		printTips();
		fgets(buffer, BUFFER_LENGTH, stdin);
		ret = sscanf(buffer, "%s %f %f %s %s", &cmd, &packetLossRatio, &ackLossRatio, &operation, &filename);
		if (!strcmp(cmd, "sr")) {
			printf("%s\n", "Begin SR protocol, please don't abort the process");
			printf("The loss ratio of packet is %.2f,the loss ratio of ack is % .2f\n", packetLossRatio, ackLossRatio);
			int waitCount = 0;
			int stage = 0;
			BOOL b;
			unsigned char u_code;//状态码
			unsigned short seq;//包的序列号
			unsigned short recvSeq;//接收窗口大小为 1，已确认的序列号
			unsigned short waitSeq;//等待的序列号
			sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
			if (!strcmp(operation, "download")) {
				//sr 0.2 0.2 download testdownload.txt
				char data[1024 * 113];
				BOOL recvd[20] = { FALSE };
				iMode = 0;
				ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & iMode);
				sendack = 0;
				while (true) {

					recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, &len);
					switch (stage) {
					case 0://等待握手阶段
						u_code = (unsigned char)buffer[0];
						if ((unsigned char)buffer[0] == 205) {
							printf("Ready for file transmission\n");
							buffer[0] = 200;
							buffer[1] = '\0';
							sendto(socketClient, buffer, 2, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
							stage = 1;
							recvSeq = 0;
							waitSeq = 0;
							totalSeq = 0;
							loct = -2;
						}
						break;
					case 1://等待接收数据阶段
						seq = (unsigned short)buffer[0];
						//随机法模拟包是否丢失
						b = lossInLossRatio(packetLossRatio);
						if (b) {
							printf("The packet with a seq of %d loss\n", seq - 1);
							continue;
						}
						printf("recv a packet with a seq of %d\n", seq - 1);
						//如果是期待的包，正确接收，正常确认即可
						seq -= 1;
						if (!(waitSeq - seq)) {
							recvd[waitSeq] = TRUE;
							memcpy(data + 1024 * totalSeq, buffer + 2, 1024);
							if (buffer[1] == '0') loct = totalSeq;
							int cnt = 10;
							while (cnt--) {
								if (recvd[waitSeq]) {
									recvd[waitSeq] = FALSE;
									++waitSeq;
									++totalSeq;
									if (waitSeq == 20) waitSeq = 0;
								}
								else break;
							}
						}
						else {
							int index = (seq + SEQ_SIZE - waitSeq) % SEQ_SIZE;
							if (index < 10 && !recvd[seq]) {
								recvd[seq] = TRUE;
								// 从buffer + 2的偏移量开始复制1024个字节到data中
								memcpy(data + 1024 * (totalSeq + index), buffer + 2, 1024);
								if (buffer[1] == '0') loct = totalSeq + index;
							}
						}
						buffer[0] = (char)(seq + 1);
						buffer[2] = '\0';
						b = lossInLossRatio(ackLossRatio);
						if (b) {
							printf("The ack of %d loss\n", (unsigned char)buffer[0] - 1);
							continue;
						}
						++sendack;
						sendto(socketClient, buffer, 3, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
						printf("send a ack of %d\n", (unsigned char)buffer[0] - 1);
						break;
					}
					if (sendack == loct + 1) {
						printf("接收完成\n");
						break;
					}
					Sleep(20);
				}
				char buff[1300];
				ofstream ofs;
				ofs.open(filename, ios::out);
				for (int i = 0; i <= loct; ++i) {
					memcpy(buff, data + 1024 * i, 1024);
					ofs << buff << endl;
				}
				ofs.close();
				if (sendack == loct + 1) {
					ZeroMemory(buffer, sizeof(buffer));
					continue;
				}
			}
			else if (!(strcmp(operation, "upload"))) {
				iMode = 1;
				ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & iMode);
				std::ifstream fin;
				fin.open(filename, ios_base::in);
				if (!fin.is_open()) {
					printf("无法打开文件");
					continue;
				}
				char buff[1024] = { 0 };
				char data[1024 * 113];
				loct = 0;
				while (fin.getline(buff, sizeof(buff))) {
					if (buff[0] == '0') break;
					memcpy(data + 1024 * loct, buff, 1024);
					++loct;
				}
				fin.close();//read file
				totalPacket = loct;
				ZeroMemory(buffer, sizeof(buffer));
				int recvSize;
				int waitCounts[21] = { 0 };
				waitCount = 0;
				printf("Begain to test SR protocol,please don't abort the process\n");

				printf("Shake hands stage\n");
				int stage = 0;
				bool runFlag = true;
				while (runFlag) {
					switch (stage) {
					case 0://发送 205 阶段
						buffer[0] = 205;
						sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
						Sleep(100);
						stage = 1;
						break;
					case 1://等待接收 200 阶段，没有收到则计数器+1，超时则放弃此次“连接”，等待从第一步开始
						recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
						if (recvSize < 0) {
							++waitCount;
							if (waitCount > 20) {
								runFlag = false;
								printf("Timeout error\n");
								break;
							}
							Sleep(20);
							continue;
						}
						else {
							if ((unsigned char)buffer[0] == 200) {
								printf("Begin a file transfer\n");
								printf("File size is %dB, each packet is 1024B and packet total num is % d\n", totalPacket * 1024, totalPacket);
								curSeq = 0;
								curAck = 0;
								totalSeq = 0;
								waitCount = 0;
								waitSeq = 0;
								stage = 2;
								for (int i = 0; i < SEQ_SIZE; ++i) {
									ack[i] = TRUE;
								}
							}
						}
						break;
					case 2:
						if (seqIsAvailable() && totalSeq < loct) {
							//发送给客户端的序列号从 1 开始
							buffer[0] = curSeq + 1;
							if (totalSeq == loct - 1) buffer[1] = '0';
							else buffer[1] = '1';
							ack[curSeq] = FALSE;
							memcpy(&buffer[2], data + 1024 * totalSeq, 1024);
							printf("send a packet with a seq of %d\n", curSeq);
							sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
							++curSeq;
							curSeq %= SEQ_SIZE;
							++totalSeq;
							Sleep(20);
						}
						//等待 Ack，若没有收到，则返回值为-1，计数器+1
						recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
						if (recvSize >= 0) {
							//收到 ack
							int i = (int)(buffer[0]) - 1;
							ack[i] = TRUE;
							waitCounts[i] = 0;
							printf("Recv a ack of %d\n", i);
							if (i == curAck) {
								if (curSeq < curAck) {
									for (; curAck < SEQ_SIZE;) {
										if (ack[curAck]) ++curAck;
										else break;
									}
									if (curAck == SEQ_SIZE) {
										for (curAck = 0; curAck < curSeq;) {
											if (ack[curAck]) ++curAck;
											else break;
										}
									}
								}
								else {
									for (; curAck < curSeq;) {
										if (ack[curAck]) ++curAck;
										else break;
									}
								}
							}
							if (curAck == curSeq && totalSeq == loct) break;
						}
						int index;
						//time out
						for (int i = 0; i < (curSeq - curAck + SEQ_SIZE) % SEQ_SIZE; ++i) {
							index = (i + curAck) % SEQ_SIZE;
							if (!ack[index]) {
								++waitCounts[index];
								if (waitCounts[index] > 20) {
									buffer[0] = index + 1;
									if (totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE) + i == loct - 1) buffer[1] = '0';
									else buffer[1] = '1';
									memcpy(&buffer[2], data + 1024 * (totalSeq - ((curSeq - curAck + SEQ_SIZE) % SEQ_SIZE) + i), 1024);
									printf("send a packet with a seq of %d\n", index);
									sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
									waitCounts[index] = 0;
								}
							}
						}
						Sleep(20);
						break;
					}
					if (curAck == curSeq && totalSeq == loct) break;
				}
				if (curAck == curSeq && totalSeq == loct) {
					printf("传输完成\n");
					ZeroMemory(buffer, sizeof(buffer));
					continue;
				}
			}
		}
		else if (!strcmp(cmd, "gbn")) {
			printf("%s\n", "Begin GBN protocol, please don't abort the process");
			printf("The loss ratio of packet is %.2f,the loss ratio of ack is % .2f\n", packetLossRatio, ackLossRatio);
			int waitCount = 0;
			int stage = 0;
			BOOL b;
			unsigned char u_code;//状态码
			unsigned short seq;//包的序列号
			unsigned short recvSeq;//接收窗口大小为 1，已确认的序列号
			unsigned short waitSeq;//等待的序列号
			unsigned short recvPacket;
			sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
			if (!strcmp(operation, "download")) {
				char data[1024 * 113];
				loct = 0;
				iMode = 0;
				int flg = 1;
				ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & iMode);
				while (true) {
					//等待 server 回复设置 UDP 为阻塞模式
					// 参数0的作用 TODO:
					recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, &len);
					switch (stage) {
					case 0://等待握手阶段
						u_code = (unsigned char)buffer[0];
						if ((unsigned char)buffer[0] == 205) {
							printf("Ready for file transmission\n");
							buffer[0] = 200;
							buffer[1] = '\0';
							sendto(socketClient, buffer, 2, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
							stage = 1;
							recvSeq = 0;
							waitSeq = 1;
							loct = 0;
						}
						break;
					case 1://等待接收数据阶段
						seq = (unsigned short)buffer[0];
						//模拟随机丢包
						b = lossInLossRatio(packetLossRatio);
						if (b) {
							printf("The packet with a seq of %d loss\n", seq - 1);
							continue;
						}
						printf("recv a packet with a seq of %d\n", seq - 1);
						//如果是期待的包，正确接收，正常确认即可
						if (!(waitSeq - seq)) {
							memcpy(data + 1024 * loct, buffer + 2, 1024);
							if (buffer[1] == '0') flg = 0;
							++loct;
							++waitSeq;
							if (waitSeq == 21) {
								waitSeq = 1;
							}
							buffer[0] = seq;
							recvSeq = seq;
							recvPacket = (unsigned short)buffer[1];
							buffer[2] = '\0';
						}
						else {
							//如果当前一个包都没有收到，则等待 Seq 为 1 的数据包，不是则不返回 ACK（因为并没有上一个正确的 ACK）
							if (!recvSeq) {
								continue;
							}
							buffer[0] = recvSeq;
							buffer[1] = recvPacket;
							buffer[2] = '\0';
						}
						b = lossInLossRatio(ackLossRatio);
						if (b) {
							printf("The ack of %d loss\n", (unsigned char)buffer[0] - 1);
							continue;
						}
						sendto(socketClient, buffer, 3, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
						printf("send a ack of %d\n", (unsigned char)buffer[0] - 1);
						break;
					}
					if (flg == 0) {
						printf("接收完成\n");
						break;
					}
					Sleep(20);
				}
				ofstream ofs;
				ofs.open(filename, ios::out);
				char buff[1300];
				printf("%d", loct);
				for (int i = 0; i < loct; ++i) {
					memcpy(buff, data + 1024 * i, 1024);
					ofs << buff << endl;
				}
				ofs.close();
				if (flg == 0) {
					ZeroMemory(buffer, sizeof(buffer));
					continue;
				}
			}
			else if (!strcmp(operation, "upload")) {
				std::ifstream fin;
				fin.open(filename, ios_base::in);
				if (!fin.is_open()) {
					printf("无法打开文件");
					continue;
				}
				iMode = 1;
				ioctlsocket(socketClient, FIONBIO, (u_long FAR*) & iMode);
				char buff[1024] = { 0 };
				char data[1024 * 113];
				loct = 0;
				int flg = 1;
				while (fin.getline(buff, sizeof(buff))) {
					if (buff[0] == '0') break;
					memcpy(data + 1024 * loct, buff, 1024);
					++loct;
				}
				fin.close();
				totalPacket = loct;
				ZeroMemory(buffer, sizeof(buffer));
				int recvSize;
				waitCount = 0;
				printf("Begain to test GBN protocol,please don't abort the process\n");
				//加入了一个握手阶段
				//首先服务器向客户端发送一个 205 大小的状态码表示服务器准备好了，可以发送数据
					//客户端收到 205 之后回复一个 200 大小的状态码，表示客户端准备好了，可以接收数据了
					//服务器收到 200 状态码之后，就开始使用 GBN 发送数据了
				printf("Shake hands stage\n");
				int stage = 0;
				bool runFlag = true;
				while (runFlag) {
					switch (stage) {
					case 0://发送 205 阶段
						buffer[0] = 205;
						sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
						Sleep(100);
						stage = 1;
						break;
					case 1://等待接收 200 阶段，没有收到则计数器+1，超时则放弃此次“连接”，等待从第一步开始
						recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
						if (recvSize < 0) {
							++waitCount;
							if (waitCount > 20) {
								runFlag = false;
								printf("Timeout error\n");
								break;
							}
							Sleep(20);
							continue;
						}
						else {
							if ((unsigned char)buffer[0] == 200) {
								printf("Begin a file transfer\n");
								printf("File size is %dB, each packet is 1024B and packet total num is % d\n", totalPacket * 1024, totalPacket);
								//准备传输，初始化
								curSeq = 0;
								curAck = 0;
								totalSeq = 0;
								waitCount = 0;
								stage = 2;
								for (int i = 0; i < SEQ_SIZE; ++i) {
									ack[i] = TRUE;
								}
							}
						}
						break;
					case 2://数据传输阶段
						if (seqIsAvailable() && totalSeq < loct) {
							//发送给客户端的序列号从 1 开始
							buffer[0] = curSeq + 1;
							if (totalSeq == loct - 1) buffer[1] = '0';
							else buffer[1] = '1';
							ack[curSeq] = FALSE;
							memcpy(&buffer[2], data + 1024 * totalSeq, 1024);
							printf("send a packet with a seq of %d\n", curSeq);
							sendto(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
							++curSeq;
							curSeq %= SEQ_SIZE;
							++totalSeq;
							Sleep(20);
						}
						//等待 Ack，若没有收到，则返回值为-1，计数器+1
						recvSize = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, ((SOCKADDR*)&addrServer), &len);
						if (recvSize < 0) {
							waitCount++;
							//20 次等待 ack 则超时重传
							if (waitCount > 20) {
								timeoutHandler();
								waitCount = 0;
							}
						}
						else {
							//收到 ack
							if (buffer[1] == '0')
							{
								flg = 0;
								break;
							}
							ackHandler(buffer[0]);
							waitCount = 0;
						}
						Sleep(20);
						break;
					}
					if (flg == 0) break;
				}
				if (flg == 0) {
					printf("传输完成\n");
					ZeroMemory(buffer, sizeof(buffer));
					continue;
				}
			}
		}
		sendto(socketClient, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
		ret = recvfrom(socketClient, buffer, BUFFER_LENGTH, 0, (SOCKADDR*)&addrServer, &len);
		printf("%s\n", buffer);
		if (!strcmp(buffer, "Good bye!")) {
			break;
		}
	}
	//关闭套接字
	closesocket(socketClient);
	WSACleanup();
	return 0;
}
