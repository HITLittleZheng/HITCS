#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/if_ether.h>
#include <sys/socket.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <linux/if_arp.h>
#include <netpacket/packet.h>
#include <errno.h>
#define BUFFER_SIZE 65536

struct route_entry {
    uint32_t dest;
    uint32_t gateway;
    uint32_t netmask;
    char interface[IFNAMSIZ];
};

struct route_entry route_table[] = {
    {inet_addr("192.168.1.0"), inet_addr("0.0.0.0"), inet_addr("255.255.255.0"), "enp0s3"},
    {inet_addr("192.168.2.0"), inet_addr("0.0.0.0"), inet_addr("255.255.255.0"), "enp0s8"}};
int route_table_size = sizeof(route_table) / sizeof(route_table[0]);

unsigned short checksum(unsigned short *buf, int nwords) {
    unsigned long sum;
    for (sum = 0; nwords > 0; nwords--)
        sum += *buf++;
    sum = (sum >> 16) + (sum & 0xffff);
    sum += (sum >> 16);
    return (unsigned short)(~sum);
}

struct route_entry *lookup_route(uint32_t dest_ip) {
    for (int i = 0; i < route_table_size; i++) {
        if ((dest_ip & route_table[i].netmask) == (route_table[i].dest & route_table[i].netmask)) {
            return &route_table[i];
        }
    }
    return NULL;
}

int get_mac_address(int sockfd, const char *iface, uint32_t dest_ip, unsigned char *mac) {
    struct arpreq req;
    memset(&req, 0, sizeof(req));
    struct sockaddr_in *sin = (struct sockaddr_in *)&req.arp_pa;
    sin->sin_family = AF_INET;
    sin->sin_addr.s_addr = dest_ip;
    strncpy(req.arp_dev, iface, IFNAMSIZ - 1);

    if (ioctl(sockfd, SIOCGARP, &req) < 0) {
        perror("ioctl SIOCGARP");
        return -1;
    }

    if (req.arp_flags & ATF_COM) {
        memcpy(mac, req.arp_ha.sa_data, ETH_ALEN);
        return 0;
    } else {
        fprintf(stderr, "ARP entry not found for %s\n", inet_ntoa(sin->sin_addr));
        return -1;
    }
}

int main() {
    int sockfd, ioctl_sockfd;
    struct sockaddr saddr;
    socklen_t saddr_len;
    unsigned char *buffer = (unsigned char *)malloc(BUFFER_SIZE);

    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP));
    if (sockfd < 0) {
        perror("Socket creation failed");
        return 1;
    }

    ioctl_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (ioctl_sockfd < 0) {
        perror("Failed to create ioctl socket");
        close(sockfd);
        return 1;
    }

    while (1) {
        saddr_len = sizeof(saddr);
        int data_size = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, &saddr, &saddr_len);
        if (data_size < 0) {
            perror("Recvfrom error");
            return 1;
        }

        struct ethhdr *eth_header = (struct ethhdr *)buffer;
        struct iphdr *ip_header = (struct iphdr *)(buffer + sizeof(struct ethhdr));

        // Ignore packets not meant for forwarding (e.g., packets with TTL <= 1)
        if (ip_header->ttl <= 1) {
            continue;
        }

        printf("Captured packet from %s to %s\n", inet_ntoa(*(struct in_addr *)&ip_header->saddr),
               inet_ntoa(*(struct in_addr *)&ip_header->daddr));

        struct route_entry *route = lookup_route(ip_header->daddr);
        if (route == NULL) {
            fprintf(stderr, "No route to host\n");
            continue;
        }

        // Decrement TTL and recalculate checksum
        ip_header->ttl -= 1;
        ip_header->check = 0;
        ip_header->check = checksum((unsigned short *)ip_header, ip_header->ihl * 2);

        // Get outgoing interface index and MAC address
        struct ifreq ifr;
        memset(&ifr, 0, sizeof(ifr));
        strncpy(ifr.ifr_name, route->interface, IFNAMSIZ - 1);
        if (ioctl(ioctl_sockfd, SIOCGIFINDEX, &ifr) < 0) {
            perror("ioctl SIOCGIFINDEX");
            continue;
        }
        int if_index = ifr.ifr_ifindex;

        // Get interface MAC address
        struct ifreq ifr_mac;
        memset(&ifr_mac, 0, sizeof(ifr_mac));
        strncpy(ifr_mac.ifr_name, route->interface, IFNAMSIZ - 1);
        if (ioctl(ioctl_sockfd, SIOCGIFHWADDR, &ifr_mac) < 0) {
            perror("ioctl SIOCGIFHWADDR");
            continue;
        }
        unsigned char source_mac[ETH_ALEN];
        memcpy(source_mac, ifr_mac.ifr_hwaddr.sa_data, ETH_ALEN);

        // Determine next hop IP
        uint32_t next_hop_ip = (route->gateway != inet_addr("0.0.0.0")) ? route->gateway : ip_header->daddr;

        // Get destination MAC address
        unsigned char dest_mac[ETH_ALEN];
        if (get_mac_address(ioctl_sockfd, route->interface, next_hop_ip, dest_mac) < 0) {
            fprintf(stderr, "Failed to get MAC address for next hop\n");
            continue;
        }

        // Update Ethernet header
        memcpy(eth_header->h_dest, dest_mac, ETH_ALEN);
        memcpy(eth_header->h_source, source_mac, ETH_ALEN);
        eth_header->h_proto = htons(ETH_P_IP);

        // Prepare destination address
        struct sockaddr_ll dest_addr;
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sll_family = AF_PACKET;
        dest_addr.sll_ifindex = if_index;
        dest_addr.sll_halen = ETH_ALEN;
        memcpy(dest_addr.sll_addr, dest_mac, ETH_ALEN);

        if (sendto(sockfd, buffer, data_size, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
            perror("Sendto error");
            continue;
        }

        printf("Forwarded packet to %s via interface %s\n", inet_ntoa(*(struct in_addr *)&ip_header->daddr), route->interface);
    }

    close(ioctl_sockfd);
    close(sockfd);
    free(buffer);
    return 0;
}
