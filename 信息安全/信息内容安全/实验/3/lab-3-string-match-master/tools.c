#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // 添加这个头文件
#include <sys/types.h>
#include <unistd.h>

void print_memory_usage() {
    char stat_path[64];
    char buffer[1024];
    FILE *stat_file;

    pid_t pid = getpid();

    // 构建 /proc/[pid]/stat 路径
    snprintf(stat_path, sizeof(stat_path), "/proc/%d/stat", pid);

    // 打开文件
    stat_file = fopen(stat_path, "r");
    if (stat_file == NULL) {
        perror("fopen");
        return;
    }

    // 读取文件内容
    if (fgets(buffer, sizeof(buffer), stat_file) == NULL) {
        perror("fgets");
        fclose(stat_file);
        return;
    }

    // 关闭文件
    fclose(stat_file);

    // 解析第 23 个字段，它是 "vsize"（虚拟内存使用量），即虚拟内存大小
    char *token = strtok(buffer, " ");
    for (int i = 1; i < 23; i++) {
        token = strtok(NULL, " ");
    }

    // 解析到第 23 个字段并输出
    if (token != NULL) {
        long vsize = atol(token); // vsize 以字节为单位
        printf("进程 %d 的虚拟内存使用 (VSize): %.2f MB\n", pid, vsize / (1024.0 * 1024.0));
    } else {
        printf("无法解析进程 %d 的虚拟内存使用情况\n", pid);
    }
}