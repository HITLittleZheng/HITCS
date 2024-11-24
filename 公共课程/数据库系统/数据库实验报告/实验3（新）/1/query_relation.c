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

// 查询并存储满足条件的元组
void query_relation_with_condition(Buffer *buf, Relation *rel, int attr_index, int query_value, unsigned int result_start_addr) {
    unsigned char *block;
    unsigned char *result_block = NULL;
    unsigned int current_addr = rel->start_addr;
    unsigned int result_addr = result_start_addr;
    int result_count = 0;

    printf("Query results for %s = %d:\n", rel->attr_names[attr_index], query_value);
    printf("+----------+----------+\n");
    printf("| %s        | %s        |\n", rel->attr_names[0], rel->attr_names[1]);
    printf("+----------+----------+\n");

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

            if (attr_value == query_value) {
                // 输出到控制台
                int attr1, attr2;
                memcpy(&attr1, ptr, sizeof(int));
                memcpy(&attr2, ptr + 4, sizeof(int));
                printf("| %-8d | %-8d |\n", attr1, attr2);

                // 如果结果块为空或已满，分配新块
                if (result_block == NULL || result_count == (BLOCK_SIZE - sizeof(int)) / 8) {   // result_count ->[0,(BLOCK_SIZE - sizeof(int)) / 8 - 1] 
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
                    result_count = 0;
                }

                // 将满足条件的元组写入结果块
                memcpy(result_block + result_count * 8, ptr, 8);
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

    printf("+----------+----------+\n");
    printf("Query results written to disk starting from block %d\n", result_start_addr);
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
        printf("Select relation (R/S), attribute (e.g., A/B/C/D), and value (e.g., 10), or 'exit': ");
        fgets(input, sizeof(input), stdin);

        // 去除换行符
        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "exit") == 0) {
            break;
        }

        char rel_name, attr_name;
        int query_value, result_start_addr;
        if (sscanf(input, "%c %c %d %d", &rel_name, &attr_name, &query_value, &result_start_addr) != 4) {
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

        query_relation_with_condition(&buf, rel, attr_index, query_value, result_start_addr);
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

// gcc -o query_relation query_relation.c extmem.c
// ./query_relation
// R A 10 200
// S C 30 201
// od -t dI -An 200.blk
// od -t dI -An 201.blk