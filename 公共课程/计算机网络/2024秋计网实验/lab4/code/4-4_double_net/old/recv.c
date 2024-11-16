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

#define BUFFER_SIZE 65536
#define PORT        12345

int main() {
    int raw_sockfd, udp_sockfd;
    struct sockaddr saddr;
    socklen_t saddr_len = sizeof(saddr);
    unsigned char buffer[BUFFER_SIZE];

    // Create raw socket
    raw_sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (raw_sockfd < 0) {
        perror("Raw socket creation failed");
        return 1;
    }

    // Create UDP socket for sending replies
    udp_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sockfd < 0) {
        perror("UDP socket creation failed");
        close(raw_sockfd);
        return 1;
    }

    while (1) {
        // Receive packet
        int data_size = recvfrom(raw_sockfd, buffer, BUFFER_SIZE, 0, &saddr, &saddr_len);
        if (data_size < 0) {
            perror("Recvfrom failed");
            return 1;
        }

        struct ethhdr *eth_header = (struct ethhdr *)buffer;                                                  // Ethernet header
        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));                           // IP header
        struct udphdr *udp_header = (struct udphdr *)(buffer + sizeof(struct ethhdr) + ip_header->ihl * 4);   // UDP header

        // Only process UDP packets destined to PORT
        if (ip_header->protocol == IPPROTO_UDP && ntohs(udp_header->dest) == PORT) {
            char *payload = (char *)(buffer + sizeof(struct ethhdr) + ip_header->ihl * 4 + sizeof(struct udphdr));
            int payload_size = data_size - (sizeof(struct ethhdr) + ip_header->ihl * 4 + sizeof(struct udphdr));
            printf("Received message: %.*s\n", payload_size, payload);

            // Print packet details
            struct in_addr saddr_in, daddr_in;
            saddr_in.s_addr = ip_header->saddr;
            daddr_in.s_addr = ip_header->daddr;

            printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
                   eth_header->h_source[0], eth_header->h_source[1], eth_header->h_source[2],
                   eth_header->h_source[3], eth_header->h_source[4], eth_header->h_source[5]);
            printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
                   eth_header->h_dest[0], eth_header->h_dest[1], eth_header->h_dest[2],
                   eth_header->h_dest[3], eth_header->h_dest[4], eth_header->h_dest[5]);
            printf("Source IP: %s\n", inet_ntoa(saddr_in));
            printf("Destination IP: %s\n", inet_ntoa(daddr_in));
            printf("TTL: %d\n", ip_header->ttl);
            printf("Source Port: %d\n", ntohs(udp_header->source));
            printf("Destination Port: %d\n", ntohs(udp_header->dest));

            // Send reply
            struct sockaddr_in dest_addr;
            memset(&dest_addr, 0, sizeof(dest_addr));
            dest_addr.sin_family = AF_INET;
            dest_addr.sin_port = udp_header->source;
            dest_addr.sin_addr.s_addr = ip_header->saddr;

            char reply_msg[1024];
            snprintf(reply_msg, sizeof(reply_msg), "Reply from server");

            if (sendto(udp_sockfd, reply_msg, strlen(reply_msg), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
                perror("Sendto failed");
            } else {
                printf("Sent reply to %s:%d\n", inet_ntoa(dest_addr.sin_addr), ntohs(dest_addr.sin_port));
            }
        }
    }

    close(udp_sockfd);
    close(raw_sockfd);
    return 0;
}
