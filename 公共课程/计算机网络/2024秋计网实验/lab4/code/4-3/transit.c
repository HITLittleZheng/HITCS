#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/if_ether.h>
#include <netinet/ether.h>
#include <sys/socket.h>
#include <unistd.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <time.h>

#define BUFFER_SIZE 65536
#define ROUTE_TABLE_SIZE 10
typedef struct {
    char dest_ip[INET_ADDRSTRLEN];
    char next_hop_ip[INET_ADDRSTRLEN];
    unsigned char next_hop_mac[ETH_ALEN];
} route_entry;

route_entry route_table[ROUTE_TABLE_SIZE] = {
    {"192.168.2.31", "192.168.2.31", {0x08, 0x00, 0x27, 0x61, 0xa8, 0x0f}},
    // 其他路由表项可以在这里添加
};

unsigned short checksum(void *b, int len) {
    unsigned short *buf = b;
    unsigned int sum = 0;
    unsigned short result;

    for (sum = 0; len > 1; len -= 2)
        sum += *buf++;
    if (len == 1)
        sum += *(unsigned char *)buf;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    result = ~sum;
    return result;
}

void print_packet_info(struct ethhdr *eth_header, struct iphdr *ip_header, const char *src_ip, const char *dest_ip) {
    char src_mac[18], dest_mac[18];
    snprintf(src_mac, sizeof(src_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eth_header->h_source[0], eth_header->h_source[1], eth_header->h_source[2], eth_header->h_source[3], eth_header->h_source[4], eth_header->h_source[5]);
    snprintf(dest_mac, sizeof(dest_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eth_header->h_dest[0], eth_header->h_dest[1], eth_header->h_dest[2], eth_header->h_dest[3], eth_header->h_dest[4], eth_header->h_dest[5]);

    time_t current_time = time(NULL);
    if (current_time == -1) {
        perror("time");
        return 1;
    }

    printf("Time: %s", ctime(&current_time));
    printf("Source MAC: %s\n", src_mac);
    printf("Destination MAC: %s\n", dest_mac);
    printf("Source IP: %s\n", src_ip);
    printf("Destination IP: %s\n", dest_ip);
    printf("TTL: %d\n", ip_header->ttl);
}

route_entry *find_route(const char *dest_ip) {
    for (int i = 0; i < ROUTE_TABLE_SIZE; i++) {
        if (strcmp(route_table[i].dest_ip, dest_ip) == 0) {
            return &route_table[i];
        }
    }
    return NULL;
}

int main() {
    int sockfd;
    struct sockaddr saddr;
    unsigned char *buffer = (unsigned char *)malloc(BUFFER_SIZE);

    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP));
    if (sockfd < 0) {
        perror("Socket creation failed");
        return 1;
    }

    while (1) {
        int saddr_len = sizeof(saddr);
        int data_size = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, &saddr, (socklen_t *)&saddr_len);
        if (data_size < 0) {
            perror("Recvfrom error");
            return 1;
        }

        struct ethhdr *eth_header = (struct ethhdr *)buffer;
        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));
        char src_ip[INET_ADDRSTRLEN];
        char dest_ip[INET_ADDRSTRLEN];

        inet_ntop(AF_INET, &(ip_header->saddr), src_ip, INET_ADDRSTRLEN);
        inet_ntop(AF_INET, &(ip_header->daddr), dest_ip, INET_ADDRSTRLEN);

        route_entry *route = find_route(dest_ip);

        if (route != NULL) {

            print_packet_info(eth_header, ip_header, src_ip, dest_ip);
            // 修改 TTL
            ip_header->ttl -= 1;
            ip_header->check = 0;
            ip_header->check = checksum((unsigned short *)ip_header, ip_header->ihl * 4);

            // 发送数据包到目的主机
            struct ifreq ifr, ifr_mac;
            struct sockaddr_ll dest;

            // 获取网卡接口索引
            memset(&ifr, 0, sizeof(ifr));
            snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "enp0s3");
            if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0) {
                perror("ioctl");
                return 1;
            }

            // 获取网卡接口 MAC 地址
            memset(&ifr_mac, 0, sizeof(ifr_mac));
            snprintf(ifr_mac.ifr_name, sizeof(ifr_mac.ifr_name), "enp0s3");
            if (ioctl(sockfd, SIOCGIFHWADDR, &ifr_mac) < 0) {
                perror("ioctl");
                return 1;
            }

            // 设置目标 MAC 地址（假设目标地址已知） 08:00:27:61:a8:0f
            // unsigned char target_mac[ETH_ALEN] = {0x08, 0x00, 0x27, 0x61, 0xa8, 0x0f}; // 替换为实际的目标 MAC 地址

            memset(&dest, 0, sizeof(dest));
            dest.sll_ifindex = ifr.ifr_ifindex;
            dest.sll_halen = ETH_ALEN;
            memcpy(dest.sll_addr, route->next_hop_mac, ETH_ALEN);

            // 构造新的以太网帧头
            memcpy(eth_header->h_dest, route->next_hop_mac, ETH_ALEN);          // 目标 MAC 地址
            memcpy(eth_header->h_source, ifr_mac.ifr_hwaddr.sa_data, ETH_ALEN); // 源 MAC地址
            eth_header->h_proto = htons(ETH_P_IP);                              // 以太网类型为 IP

            printf("Interface name: %s, index: %d\n", ifr.ifr_name, ifr.ifr_ifindex);
            if (sendto(sockfd, buffer, data_size, 0, (struct sockaddr *)&dest, sizeof(dest)) < 0) {
                perror("Sendto error");
                return 1;
            }
            printf("Datagram forwarded to next hop %s (MAC: %02x:%02x:%02x:%02x:%02x:%02x).\n", route->next_hop_ip, route->next_hop_mac[0], route->next_hop_mac[1], route->next_hop_mac[2], route->next_hop_mac[3], route->next_hop_mac[4], route->next_hop_mac[5]);
            memset(buffer, 0, BUFFER_SIZE);
            printf("================================================\n");
        } else {
            // printf("Ignored packet from %s to %s\n", src_ip, dest_ip);
        }
    }

    close(sockfd);
    free(buffer);
    return 0;
}