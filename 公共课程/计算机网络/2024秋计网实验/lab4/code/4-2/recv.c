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

#define BUFFER_SIZE 65536
#define PORT        12345

void print_packet_info(struct ethhdr *eth_header, struct iphdr *ip_header, struct udphdr *udp_header, int payload_size) {
    char src_mac[18], dest_mac[18], src_ip[INET_ADDRSTRLEN], dest_ip[INET_ADDRSTRLEN];
    snprintf(src_mac, sizeof(src_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eth_header->h_source[0], eth_header->h_source[1], eth_header->h_source[2], eth_header->h_source[3], eth_header->h_source[4], eth_header->h_source[5]);
    snprintf(dest_mac, sizeof(dest_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eth_header->h_dest[0], eth_header->h_dest[1], eth_header->h_dest[2], eth_header->h_dest[3], eth_header->h_dest[4], eth_header->h_dest[5]);
    inet_ntop(AF_INET, &ip_header->saddr, src_ip, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &ip_header->daddr, dest_ip, INET_ADDRSTRLEN);

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
    printf("Payload size: %d bytes\n", payload_size);
}

int main() {
    int sockfd;
    struct sockaddr saddr;
    socklen_t saddr_len = sizeof(saddr);
    unsigned char buffer[BUFFER_SIZE];

    // 创建原始套接字
    /**
     * AF_PACKET
     * AF_PACKET 表示使用"数据链路层"的套接字地址族。使用 AF_PACKET 可以直接访问网络接口层的数据帧，可以捕获和发送链路层的数据包（例如以太网帧）。
     * 这种套接字类型通常用于编写网络协议分析程序，如 tcpdump、wireshark 等。
     * SOCK_RAW
     *
     * SOCK_RAW 指定创建一个RAW套接字，允许接收和发送原始的网络数据包。
     * RAW套接字不会经过内核的协议栈处理，可以直接读取和处理数据帧中的数据，是一种低级网络编程方式。
     * 使用RAW套接字需要root权限，因为它可以访问和操作底层的网络数据包。
     * htons(ETH_P_ALL)
     *
     * htons 是一个函数，用于将主机字节序转换为网络字节序（大端字节序）。
     * ETH_P_ALL 是一个宏，定义在 <linux/if_ether.h> 中，表示监听所有协议类型的数据包。
     * 使用 htons(ETH_P_ALL) 表示这个套接字将接收所有类型的以太网数据包，不仅限于特定协议的数据包（例如，只接收 ARP 或 IP 数据包）。
     */
    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL)); // 接受全部协议的包
    if (sockfd < 0) {
        perror("Socket creation failed");
        return 1;
    }

    while (1) {
        // 接收数据包
        int data_size = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, &saddr, &saddr_len);
        if (data_size < 0) {
            perror("Recvfrom failed");
            return 1;
        }

        struct ethhdr *eth_header = (struct ethhdr *)buffer;                                                  // 以太网头部
        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));                           // IP头部
        struct udphdr *udp_header = (struct udphdr *)(buffer + sizeof(struct ethhdr) + sizeof(struct iphdr)); // UDP头部

        // 仅处理目标端口为 12345 的 UDP 数据包
        if (ip_header->protocol == IPPROTO_UDP && ntohs(udp_header->dest) == PORT) {
            char *payload = (char *)(buffer + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr));
            int payload_size = data_size - (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr));
            print_packet_info(eth_header, ip_header, udp_header, payload_size);
            printf("Received message: %.*s\n", payload_size, payload);
            printf("================================================");
        }
        memset(buffer, 0, BUFFER_SIZE);
    }

    close(sockfd);
    return 0;
}
