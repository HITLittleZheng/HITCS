#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/if_ether.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <net/if.h>
#include <net/ethernet.h>
#include <linux/if_packet.h>
#include <time.h>

#define BUFFER_SIZE 65536
const char *source_ip = "192.168.1.2";
struct route_entry
{
    uint32_t dest;
    uint32_t gateway;
    uint32_t netmask;
    char interface[IF_NAMESIZE];
};

struct route_entry route_table[2];

unsigned short checksum(unsigned short *buf, int nwords)
{
    unsigned long sum;
    for (sum = 0; nwords > 0; nwords--)
        sum += *buf++;
    sum = (sum >> 16) + (sum & 0xffff);
    sum += (sum >> 16);
    return (unsigned short)(~sum);
}

struct route_entry *lookup_route(uint32_t dest_ip)
{
    int route_table_size = sizeof(route_table) / sizeof(route_table[0]);
    // printf("route_table_size: %d\n", route_table_size);
    for (int i = 0; i < route_table_size; i++)
    {
        if ((dest_ip & route_table[i].netmask) == (route_table[i].dest &
                                                   route_table[i].netmask))
        {
            return &route_table[i];
        }
    }
    return NULL;
}

int main()
{
    route_table[0].dest = inet_addr("192.168.2.0");
    route_table[0].gateway = inet_addr("192.168.1.1");
    route_table[0].netmask = inet_addr("255.255.255.0");
    strncpy(route_table[0].interface, "enp0s8", IF_NAMESIZE);

    route_table[1].dest = inet_addr("192.168.1.0");
    route_table[1].gateway = inet_addr("192.168.2.1");
    route_table[1].netmask = inet_addr("255.255.255.0");
    strncpy(route_table[1].interface, "enp0s3", IF_NAMESIZE);

    int sockfd;
    struct sockaddr saddr;
    unsigned char *buffer = (unsigned char *)malloc(BUFFER_SIZE);
    if (buffer == NULL)
    {
        perror("Memory allocation failed");
        return 1;
    }
    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP));
    if (sockfd < 0)
    {
        perror("Socket creation failed");
        return 1;
    }

    while (1)
    {
        int saddr_len = sizeof(saddr);
        int data_size = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, &saddr, (socklen_t *)&saddr_len);
        if (data_size < 0)
        {
            perror("Recvfrom error");
            return 1;
        }

        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));
        // 获取当前系统时间
        time_t rawtime;
        struct tm *timeinfo;
        char time_str[100];

        time(&rawtime);
        timeinfo = localtime(&rawtime);

        // 格式化时间字符串
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);

        // 打印信息
        printf("[%s] Captured packet\n", time_str);
        printf("Message from %s\n", inet_ntoa(*(struct in_addr *)&ip_header->saddr));
        printf("Message will send to %s\n", inet_ntoa(*(struct in_addr *)&ip_header->daddr));
        struct route_entry *route = lookup_route(ip_header->daddr);
        if (route == NULL)
        {
            fprintf(stderr, "found no host in routes_map\n");
            continue;
        }

        // 修改 TTL
        ip_header->ttl -= 1;
        ip_header->check = 0;
        ip_header->check = checksum((unsigned short *)ip_header, ip_header->ihl * 4);

        // 发送数据包到目的主机
        struct ifreq ifr, ifr_mac;
        struct sockaddr_ll dest;

        // 获取网卡接口索引
        memset(&ifr, 0, sizeof(ifr));
        snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", route->interface);
        if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0)
        {
            perror("ioctl");
            return 1;
        }

        // 获取网卡接口 MAC 地址
        memset(&ifr_mac, 0, sizeof(ifr_mac));
        snprintf(ifr_mac.ifr_name, sizeof(ifr_mac.ifr_name), "%s", route->interface);
        if (ioctl(sockfd, SIOCGIFHWADDR, &ifr_mac) < 0)
        {
            perror("ioctl");
            return 1;
        }

        // 设置目标 MAC 地址（假设目标地址已知,此处做了简化处理，实际上，如果查找路由表后，存在“下
        // 一跳”，应该利用 ARP 协议获得 route->gateway 的 MAC 地址，如果是“直接交付”的话，也应使用 ARP 协议获得
        //     目的主机的 MAC 地址。） unsigned char target_mac[ETH_ALEN] = {0x00, 0x0c, 0x29, 0x48, 0xd3, 0xf7}; //
        // 替换为实际的目标 MAC 地址
        char ip_str[10];
        strncpy(ip_str, inet_ntoa(*(struct in_addr *)&ip_header->saddr), 9);
        ip_str[9] = '\0';
        if (strcmp(ip_str, "192.168.2") == 0)
        {
            printf("Router send to enp0s3\n");
            printf("Source Mac: %s\n", "08:00:27:87:fb:a7");
            printf("Dest Mac: %s\n", "08:00:27:c7:a7:1a");

            printf("former TTL: %d\n", ip_header->ttl + 1);
            printf("current TTL: %d\n", ip_header->ttl);
            unsigned char target_mac[ETH_ALEN] = {0x08, 0x00, 0x27, 0xc7, 0xa7, 0x1a}; // 替换为实际的目标 MAC 地址
            memset(&dest, 0, sizeof(dest));
            dest.sll_ifindex = ifr.ifr_ifindex;
            dest.sll_halen = ETH_ALEN;
            memcpy(dest.sll_addr, target_mac, ETH_ALEN);

            // 构造新的以太网帧头
            struct ethhdr *eth_header = (struct ethhdr *)buffer;
            memcpy(eth_header->h_dest, target_mac, ETH_ALEN);                   // 目标 MAC 地址
            memcpy(eth_header->h_source, ifr_mac.ifr_hwaddr.sa_data, ETH_ALEN); // 源 MAC 地址
            eth_header->h_proto = htons(ETH_P_IP);                              // 以太网类型为 IP

            printf("Interface name: %s, index: %d\n", ifr.ifr_name, ifr.ifr_ifindex);
        }
        else if (strcmp(ip_str, "192.168.1") == 0)
        {
            printf("Router send to enp0s8\n");
            printf("Source Mac: %s\n", "08:00:27:c7:a7:1a");
            printf("Dest Mac: %s\n", "08:00:27:87:fb:a7");
            printf("former TTL: %d\n", ip_header->ttl + 1);
            printf("current TTL: %d\n", ip_header->ttl);
            unsigned char target_mac[ETH_ALEN] = {0x08, 0x00, 0x27, 0x87, 0xfb, 0xa7}; // 替换为实际的目标 MAC 地址
            memset(&dest, 0, sizeof(dest));
            dest.sll_ifindex = ifr.ifr_ifindex;
            dest.sll_halen = ETH_ALEN;
            memcpy(dest.sll_addr, target_mac, ETH_ALEN);

            // 构造新的以太网帧头
            struct ethhdr *eth_header = (struct ethhdr *)buffer;
            memcpy(eth_header->h_dest, target_mac, ETH_ALEN);                   // 目标 MAC 地址
            memcpy(eth_header->h_source, ifr_mac.ifr_hwaddr.sa_data, ETH_ALEN); // 源 MAC 地址
            eth_header->h_proto = htons(ETH_P_IP);                              // 以太网类型为 IP

            printf("Interface name: %s, index: %d\n", ifr.ifr_name, ifr.ifr_ifindex);
        }

        if (sendto(sockfd, buffer, data_size, 0, (struct sockaddr *)&dest, sizeof(dest)) < 0)
        {
            perror("Sendto error");
            return 1;
        }
    }

    close(sockfd);
    free(buffer);
    return 0;
}