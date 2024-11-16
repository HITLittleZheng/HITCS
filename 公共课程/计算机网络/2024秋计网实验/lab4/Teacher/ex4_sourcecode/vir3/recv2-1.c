#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <linux/if_packet.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <time.h>
#define PORT 12345
#define BUFFER_SIZE 65536
void get_time(char *time_str);
int main()
{
    int sockfd;
    struct sockaddr saddr;
    socklen_t saddr_len = sizeof(saddr);
    unsigned char buffer[BUFFER_SIZE];
    char time_str[20];
    // 创建原始套接字
    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0)
    {
        perror("Socket creation failed");
        return 1;
    }

    while (1)
    {
        // 接收数据包
        int data_size = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, &saddr, &saddr_len);
        if (data_size < 0)
        {
            perror("Recvfrom failed");
            return 1;
        }

        struct ethhdr *eth_header = (struct ethhdr *)buffer;                                                  // 以太网头部
        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));                           // IP头部
        struct udphdr *udp_header = (struct udphdr *)(buffer + sizeof(struct ethhdr) + sizeof(struct iphdr)); // UDP头部

        // 仅处理目标端口为 12345 的 UDP 数据包
        if (ip_header->protocol == IPPROTO_UDP && ntohs(udp_header->dest) == PORT)
        {
            get_time(time_str);
            if (ip_header->ttl == 254)
            {
                printf("收到时间: %s\n", time_str);
                char *payload = (char *)(buffer + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr));
                int payload_size = data_size - (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr));
                printf("Received message: %.*s\n", payload_size, payload);
                printf("Source IP: %s\n", inet_ntoa(*(struct in_addr *)&ip_header->saddr));
                printf("Destination IP: %s\n", inet_ntoa(*(struct in_addr *)&ip_header->daddr));
                printf("Source mac: %02x:%02x:%02x:%02x:%02x:%02x\n", eth_header->h_source[0], eth_header->h_source[1],
                       eth_header->h_source[2], eth_header->h_source[3], eth_header->h_source[4], eth_header->h_source[5]);
                printf("Destination mac: %02x:%02x:%02x:%02x:%02x:%02x\n", eth_header->h_dest[0], eth_header->h_dest[1],
                       eth_header->h_dest[2], eth_header->h_dest[3], eth_header->h_dest[4], eth_header->h_dest[5]);
                printf("TTL: %d\n", ip_header->ttl);
            }
        }
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
