#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>

void get_time(char *time_str);

int main()
{
    int sockfd;
    struct sockaddr_in dest_addr;
    char message[1024];
    int port = 12345; // 目标端口号
    char time_str[20];
    // 创建 UDP 套接字
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        perror("socket");
        return 1;
    }

    // 目标地址
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);
    dest_addr.sin_addr.s_addr = inet_addr("192.168.18.160");

    while (1)
    {
        printf("从控制台输入数据，输入exit退出\n");
        memset(message, 0, sizeof(message));
        fgets(message, sizeof(message), stdin);
        message[strcspn(message, "\n")] = '\0';
        if (strcmp(message, "exit") == 0)
        {
            break;
        }
        // 发送数据报
        if (sendto(sockfd, message, strlen(message), 0, (struct sockaddr *)&dest_addr,
                   sizeof(dest_addr)) < 0)
        {
            perror("sendto");
            return 1;
        }
        get_time(time_str);
        printf("Datagram sent:%s\n", message);
        printf("Time:%s\n", time_str);
    }

    close(sockfd);
    return 0;
}
// 获取当前时间函数
void get_time(char *time_str)
{
    time_t timep;
    struct tm *p;
    time(&timep);
    p = localtime(&timep);
    sprintf(time_str, "%d-%d-%d %d:%d:%d", (1900 + p->tm_year), (1 + p->tm_mon), p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
}
