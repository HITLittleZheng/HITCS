#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEST_IP   "192.168.2.2"
#define DEST_PORT 12345
#define LOCAL_PORT 54321
#define MESSAGE   "Hello, this is a test message."

int main() {
    int sockfd;
    struct sockaddr_in dest_addr, local_addr;
    char buffer[1024];
    socklen_t addr_len = sizeof(struct sockaddr_in);

    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return 1;
    }

    // Bind to local port for receiving replies
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(LOCAL_PORT);
    local_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        return 1;
    }

    // Set destination address
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(DEST_PORT);
    inet_pton(AF_INET, DEST_IP, &dest_addr.sin_addr);

    // Send data packet
    if (sendto(sockfd, MESSAGE, strlen(MESSAGE), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("Sendto failed");
        close(sockfd);
        return 1;
    }

    printf("Message sent to %s:%d\n", DEST_IP, DEST_PORT);

    // Wait for reply
    struct sockaddr_in from_addr;
    int n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0, (struct sockaddr *)&from_addr, &addr_len);
    if (n < 0) {
        perror("Recvfrom failed");
    } else {
        buffer[n] = '\0';
        printf("Received reply from %s:%d: %s\n", inet_ntoa(from_addr.sin_addr), ntohs(from_addr.sin_port), buffer);
    }

    close(sockfd);
    return 0;
}
