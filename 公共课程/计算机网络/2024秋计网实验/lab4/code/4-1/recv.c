#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <time.h>
#include <string.h>
void get_time(char *time_str);

int main() {
    int sockfd;
    struct sockaddr_in src_addr, my_addr;
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    socklen_t addr_len;
    int port = 54321; // 修改后的接收端口号
    char time_str[20];

    // 创建 UDP 套接字
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
        perror("socket");
        return 1;
    }

    // 本地地址
    my_addr.sin_family = AF_INET;
    my_addr.sin_port = htons(port);
    my_addr.sin_addr.s_addr = INADDR_ANY;

    // 绑定套接字到本地地址
    if (bind(sockfd, (struct sockaddr *)&my_addr, sizeof(my_addr)) < 0) {
        perror("bind");
        return 1;
    }

    while (1) {
        // 接收数据报
        addr_len = sizeof(src_addr);
        if (recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr *)&src_addr, &addr_len) < 0) {
            perror("recvfrom");
            return 1;
        }

        get_time(time_str);
        printf("Datagram received at %s\n", time_str);
        printf("Source Port: %d\n", ntohs(src_addr.sin_port));
        printf("Destination Port: %d\n", port);
        printf("Message: %s\n", buffer);
        memset(buffer, 0, sizeof(buffer));
    }

    close(sockfd);
    return 0;
}

// 获取当前时间函数
void get_time(char *time_str) {
    time_t timep;
    struct tm *p;
    time(&timep);
    p = localtime(&timep);
    sprintf(time_str, "%d-%d-%d %d:%d:%d", (1900 + p->tm_year), (1 + p->tm_mon), p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
}