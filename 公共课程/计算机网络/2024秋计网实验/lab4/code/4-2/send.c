#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <linux/if_packet.h>
#include <time.h>
// 08:00:27:c2:af:a4
#define DEST_MAC0 0x08
#define DEST_MAC1 0x00
#define DEST_MAC2 0x27
#define DEST_MAC3 0xc2
#define DEST_MAC4 0xaf
#define DEST_MAC5 0xa4

#define ETHER_TYPE   0x0800
#define BUFFER_SIZE  1518
#define UDP_SRC_PORT 12345
#define UDP_DST_PORT 12345

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

// 获取当前时间函数
void get_time(char *time_str) {
    time_t timep;
    struct tm *p;
    time(&timep);
    p = localtime(&timep);
    sprintf(time_str, "%d-%d-%d %d:%d:%d", (1900 + p->tm_year), (1 + p->tm_mon), p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
}

void print_packet_info(struct ether_header *eh, struct iphdr *iph) {
    char src_mac[18], dest_mac[18], src_ip[INET_ADDRSTRLEN], dest_ip[INET_ADDRSTRLEN];
    snprintf(src_mac, sizeof(src_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eh->ether_shost[0], eh->ether_shost[1], eh->ether_shost[2], eh->ether_shost[3], eh->ether_shost[4], eh->ether_shost[5]);
    snprintf(dest_mac, sizeof(dest_mac), "%02x:%02x:%02x:%02x:%02x:%02x", eh->ether_dhost[0], eh->ether_dhost[1], eh->ether_dhost[2], eh->ether_dhost[3], eh->ether_dhost[4], eh->ether_dhost[5]);
    inet_ntop(AF_INET, &iph->saddr, src_ip, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &iph->daddr, dest_ip, INET_ADDRSTRLEN);

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
    printf("TTL: %d\n", iph->ttl);
}

int main() {
    int sockfd;
    struct ifreq if_idx, if_mac;
    struct sockaddr_ll socket_address;
    char buffer[BUFFER_SIZE];

    // 创建原始套接字
    if ((sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) == -1) {
        perror("socket");
        return 1;
    }

    // 获取接口索引
    memset(&if_idx, 0, sizeof(struct ifreq));
    strncpy(if_idx.ifr_name, "enp0s3", IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) {
        perror("SIOCGIFINDEX");
        return 1;
    }

    // 获取接口 MAC 地址
    memset(&if_mac, 0, sizeof(struct ifreq));
    strncpy(if_mac.ifr_name, "enp0s3", IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) {
        perror("SIOCGIFHWADDR");
        return 1;
    }

    // 构造以太网头
    struct ether_header *eh = (struct ether_header *)buffer;
    eh->ether_shost[0] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[0];
    eh->ether_shost[1] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[1];
    eh->ether_shost[2] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[2];
    eh->ether_shost[3] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[3];
    eh->ether_shost[4] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[4];
    eh->ether_shost[5] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[5];
    eh->ether_dhost[0] = DEST_MAC0;
    eh->ether_dhost[1] = DEST_MAC1;
    eh->ether_dhost[2] = DEST_MAC2;
    eh->ether_dhost[3] = DEST_MAC3;
    eh->ether_dhost[4] = DEST_MAC4;
    eh->ether_dhost[5] = DEST_MAC5;
    eh->ether_type = htons(ETHER_TYPE);

    // 构造 IP 头
    struct iphdr *iph = (struct iphdr *)(buffer + sizeof(struct ether_header));
    iph->ihl = 5; // 设置IP头部长度（IHL），表示IP头部的长度为5个32位字（即20字节）。
    iph->version = 4; // 设置IP版本号为IPv4。
    iph->tos = 0; // 设置服务类型（Type of Service），这里设置为0，表示默认服务
    iph->tot_len = htons(sizeof(struct iphdr) + sizeof(struct udphdr) + strlen("Hello, this is a test message."));
    iph->id = htonl(54321); // 设置标识字段，用于唯一标识数据包，并使用htonl函数将其转换为网络字节序。
    iph->frag_off = 0; // 设置片偏移字段，这里设置为0，表示数据包没有被分片。
    iph->ttl = 255; // 设置生存时间（Time to Live），表示数据包在网络中可以经过的最大跳数，这里设置为255。
    iph->protocol = IPPROTO_UDP; // 设置协议字段，这里设置为UDP协议。
    iph->check = 0; // 初始化校验和字段为0，稍后会计算实际的校验和。
    iph->saddr = inet_addr("192.168.2.27");
    iph->daddr = inet_addr("192.168.2.31");
    iph->check = checksum((unsigned short *)iph, sizeof(struct iphdr)); // 计算并设置IP头部的校验和，使用checksum函数计算整个IP头部的校验和。

    // 构造 UDP 头
    struct udphdr *udph = (struct udphdr *)(buffer + sizeof(struct ether_header) + sizeof(struct iphdr));
    udph->source = htons(UDP_SRC_PORT);
    udph->dest = htons(UDP_DST_PORT);
    udph->len = htons(sizeof(struct udphdr) + strlen("Hello, this is a test message."));
    udph->check = 0; // UDP 校验和可选

    // 填充数据
    char *data = (char *)(buffer + sizeof(struct ether_header) + sizeof(struct iphdr) + sizeof(struct udphdr));
    strcpy(data, "Hello, this is a test message.");

    // 设置 socket 地址结构
    socket_address.sll_ifindex = if_idx.ifr_ifindex;
    socket_address.sll_halen = ETH_ALEN;
    socket_address.sll_addr[0] = DEST_MAC0;
    socket_address.sll_addr[1] = DEST_MAC1;
    socket_address.sll_addr[2] = DEST_MAC2;
    socket_address.sll_addr[3] = DEST_MAC3;
    socket_address.sll_addr[4] = DEST_MAC4;
    socket_address.sll_addr[5] = DEST_MAC5;

    // 打印数据包信息
    print_packet_info(eh, iph);

    // 发送数据包
    if (sendto(sockfd, buffer, sizeof(struct ether_header) + sizeof(struct iphdr) + sizeof(struct udphdr) + strlen("Hello, this is a test message."), 0, (struct sockaddr *)&socket_address, sizeof(struct sockaddr_ll)) < 0) {
        perror("sendto");
        return 1;
    }
    
    
    close(sockfd);
    return 0;
}