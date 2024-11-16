#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 12345
#define MESSAGE "Hello, this is a test message."
#define REPONSE "Hello! Got your message loud and clear."
#define DEST_PORT 12345
#define DEST_IP "192.168.1.2"

int main()
{
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);
    char buffer[1024];

    // 创建 UDP 套接字
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        perror("Socket creation failed");
        return 1;
    }

    // 绑定套接字到端口
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("Bind failed");
        return 1;
    }

    /////
    int sockfd_send;
    struct sockaddr_in dest_addr;

    // 创建 UDP 套接字
    sockfd_send = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_send < 0)
    {
        perror("Socket creation failed");
        return 1;
    }

    // 设置目的地址
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(DEST_PORT);
    inet_pton(AF_INET, DEST_IP, &dest_addr.sin_addr);

    while (1)
    {
        // 接收数据包
        int recv_len = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0, (struct sockaddr *)&client_addr, &addr_len);
        if (recv_len < 0)
        {
            perror("Recvfrom failed");
            return 1;
        }

        buffer[recv_len] = '\0';
        printf("Received message: %s\n", buffer);
        if (strcmp(buffer, MESSAGE) == 0) // 发送回复消息
        {
            memset(buffer, 0, sizeof(buffer));
            strncpy(buffer, REPONSE, sizeof(REPONSE));

            if (sendto(sockfd_send, REPONSE, strlen(REPONSE), 0, (struct sockaddr *)&dest_addr,
                       sizeof(dest_addr)) < 0)
            {
                perror("Sendto failed");
                return 1;
            }
            printf("Message send from %s\n", "192.168.2.2");
            printf("Message sent to %s:%d\n", DEST_IP, DEST_PORT);
        }
    }
    close(sockfd_send);
    close(sockfd);
    return 0;
}
