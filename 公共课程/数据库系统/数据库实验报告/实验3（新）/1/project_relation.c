#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extmem.h"

#define BLOCK_SIZE 64
#define BUFFER_SIZE 520

// 定义关系模式结构体
typedef struct {
    unsigned int start_addr; // 起始磁盘块地址
    int attr_count;          // 属性个数
    int attr_offsets[2];     // 属性偏移量（仅支持 2 个属性 A/B 或 C/D）
    char attr_names[2][10];  // 属性名称
} Relation;

// 投影函数：对指定属性进行投影，并将结果写入磁盘
void project_relation(Buffer *buf, Relation *rel, int attr_index, unsigned int result_start_addr) {
    unsigned char *block;
    unsigned char *result_block = NULL;
    unsigned int current_addr = rel->start_addr;
    unsigned int result_addr = result_start_addr;
    int result_count = 0;
    int result_set[1000] = {0}; // 简单使用布尔数组模拟结果集合（假设值域有限）

    printf("Projecting attribute %s on relation:\n", rel->attr_names[attr_index]);
    printf("+----------+\n");
    printf("| %s        |\n", rel->attr_names[attr_index]);
    printf("+----------+\n");

    while (current_addr != 0) {
        // 读取磁盘块到缓冲区
        block = readBlockFromDisk(current_addr, buf);
        if (!block) {
            printf("Failed to read block %d\n", current_addr);
            return;
        }

        unsigned char *ptr = block;

        // 遍历块内的元组
        for (int i = 0; i < (BLOCK_SIZE - sizeof(int)) / 8; i++) {
            int attr_value;
            memcpy(&attr_value, ptr + rel->attr_offsets[attr_index], sizeof(int)); // 获取指定属性值
            if (result_set[attr_value] == 0) { // 如果该值尚未被投影
                result_set[attr_value] = 1; // 标记为已投影

                // 输出到控制台
                printf("| %-8d |\n", attr_value);

                // 如果结果块为空或已满，分配新块
                if (result_block == NULL || result_count == (BLOCK_SIZE - sizeof(int)) / 4) {
                    if (result_block != NULL) {
                        // 写入结果块的后继地址
                        int next_addr = result_addr + 1;
                        memcpy(result_block + BLOCK_SIZE - sizeof(int), &next_addr, sizeof(int));
                        writeBlockToDisk(result_block, result_addr, buf);
                        freeBlockInBuffer(result_block, buf);
                        result_addr++;
                    }

                    // 分配新结果块
                    result_block = getNewBlockInBuffer(buf);
                    if (!result_block) {
                        printf("Buffer full while allocating result block!\n");
                        return;
                    }
                    memset(result_block, 0, BLOCK_SIZE);
                    result_count = 0; // 重置计数
                }

                // 将属性值写入结果块
                memcpy(result_block + result_count * 4, &attr_value, sizeof(int));
                result_count++;
            }

            ptr += 8; // 移动到下一个元组
        }

        // 获取当前块的后继地址
        memcpy(&current_addr, block + BLOCK_SIZE - sizeof(int), sizeof(int));
        freeBlockInBuffer(block, buf);
    }

    // 写入最后一个结果块
    if (result_block != NULL) {
        int end_addr = 0;
        memcpy(result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));
        writeBlockToDisk(result_block, result_addr, buf);
        freeBlockInBuffer(result_block, buf);
    }

    printf("+----------+\n");
    printf("Projection results written to disk starting from block %d\n", result_start_addr);
}


int main() {
    Buffer buf;
    if (!initBuffer(BUFFER_SIZE, BLOCK_SIZE, &buf)) {
        printf("Buffer initialization failed!\n");
        return -1;
    }

    // 定义关系 R 和 S
    Relation R = {1, 2, {0, 4}, {"A", "B"}}; // R 的起始地址为 1，属性为 A 和 B
    Relation S = {100, 2, {0, 4}, {"C", "D"}}; // S 的起始地址为 17，属性为 C 和 D

    char input[100];
    while (1) {
        printf("Select relation (R/S), attribute (e.g., A/B/C/D) or 'exit': ");
        fgets(input, sizeof(input), stdin);

        // 去除换行符
        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "exit") == 0) {
            break;
        }

        char rel_name, attr_name;
        int query_value, result_start_addr;
        if (sscanf(input, "%c %c %d", &rel_name, &attr_name, &result_start_addr) != 3) {
            printf("Invalid input format. Try again.\n");
            continue;
        }

        Relation *rel = NULL;
        if (rel_name == 'R' || rel_name == 'r') {
            rel = &R;
        } else if (rel_name == 'S' || rel_name == 's') {
            rel = &S;
        } else {
            printf("Invalid relation name. Use 'R' or 'S'.\n");
            continue;
        }

        int attr_index = -1;
        for (int i = 0; i < rel->attr_count; i++) {
            if (attr_name == rel->attr_names[i][0]) {
                attr_index = i;
                break;
            }
        }

        if (attr_index == -1) {
            printf("Invalid attribute name. Use one of %s or %s.\n", rel->attr_names[0], rel->attr_names[1]);
            continue;
        }
        project_relation(&buf, rel, attr_index, result_start_addr); 
    }

        // 打印 Buffer 参数为表格形式
    printf("\nBuffer Statistics:\n");
    printf("+-----------------------+--------------------+\n");
    printf("| Parameter             | Value              |\n");
    printf("+-----------------------+--------------------+\n");
    printf("| Number of I/O         | %-18lu |\n", buf.numIO);
    printf("| Buffer size (bytes)   | %-18zu |\n", buf.bufSize);
    printf("| Block size (bytes)    | %-18zu |\n", buf.blkSize);
    printf("| Total blocks          | %-18zu |\n", buf.numAllBlk);
    printf("| Free blocks           | %-18zu |\n", buf.numFreeBlk);
    printf("+-----------------------+--------------------+\n");

    freeBuffer(&buf);
    return 0;
}

// gcc -o project_relation project_relation.c extmem.c
// ./project_relation 
// R A 202
// od -t dI -An 209.blk
