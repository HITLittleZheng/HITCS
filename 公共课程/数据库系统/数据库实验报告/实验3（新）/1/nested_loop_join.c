#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extmem.h"

// 假设定义的宏和常量
#define BLOCK_SIZE 64
#define BUFFER_SIZE 520
#define MAX_TUPLES_PER_BLOCK ((BLOCK_SIZE - sizeof(int)) / 8)  // 每个块最多存储的元组数量

// 定义关系模式结构体
typedef struct {
    unsigned int start_addr; // 起始磁盘块地址
    int attr_count;          // 属性个数
    int attr_offsets[2];     // 属性偏移量（仅支持 2 个属性 A/B 或 C/D）
    char attr_names[2][10];  // 属性名称
} Relation;


// 连接操作
void nested_loop_join(Buffer *buf, Relation *R, Relation *S, unsigned int result_start_addr) {
    unsigned char *R_block, *S_block, *result_block = NULL;
    unsigned int R_addr = R->start_addr, S_addr = S->start_addr, result_addr = result_start_addr;
    int result_count = 0;
    int total_join_num = 0;

    printf("Performing Nested-Loop Join (R.A = S.C)\n");
    printf("+------+------+------+------+\n");
    printf("|  R.A |  R.B |  S.C |  S.D |\n");
    printf("+------+------+------+------+\n");

    // 遍历关系 R 的每个块
    while (R_addr != 0) {
        R_block = readBlockFromDisk(R_addr, buf);
        if (!R_block) {
            printf("Failed to read block %d of R\n", R_addr);
            return;
        }

        // 对于每个 R 的块，遍历关系 S 的所有块
        S_addr = S->start_addr;
        while (S_addr != 0) {
            S_block = readBlockFromDisk(S_addr, buf);
            if (!S_block) {
                printf("Failed to read block %d of S\n", S_addr);
                return;
            }

            // 遍历 R 块中的每个元组
            unsigned char *R_ptr = R_block;
            for (int i = 0; i < MAX_TUPLES_PER_BLOCK; i++) {
                int R_A, R_B;
                memcpy(&R_A, R_ptr, sizeof(int));    // 获取 R.A
                memcpy(&R_B, R_ptr + 4, sizeof(int)); // 获取 R.B
                R_ptr += 8;  // 移动到下一个元组

                // 遍历 S 块中的每个元组
                unsigned char *S_ptr = S_block;
                for (int j = 0; j < MAX_TUPLES_PER_BLOCK; j++) {
                    int S_C, S_D;
                    memcpy(&S_C, S_ptr, sizeof(int));    // 获取 S.C
                    memcpy(&S_D, S_ptr + 4, sizeof(int)); // 获取 S.D
                    S_ptr += 8;  // 移动到下一个元组

                    // 检查是否满足连接条件 R.A = S.C
                    if (R_A == S_C) {
                        // 连接成功，输出到结果
                        if (result_block == NULL || result_count == (BLOCK_SIZE - sizeof(int)) / 16) {
                            // 写入结果块
                            if (result_block != NULL) {
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

                        // 将连接结果写入结果块
                        memcpy(result_block + result_count * 16, &R_A, sizeof(int));  // R.A
                        memcpy(result_block + result_count * 16 + 4, &R_B, sizeof(int)); // R.B
                        memcpy(result_block + result_count * 16 + 8, &S_C, sizeof(int)); // S.C
                        memcpy(result_block + result_count * 16 + 12, &S_D, sizeof(int)); // S.D
                        result_count++;
                        total_join_num++;

                        // 打印连接结果
                        printf("| %-4d | %-4d | %-4d | %-4d |\n", R_A, R_B, S_C, S_D);
                    }
                }
            }

            // 获取下一个 S 块
            memcpy(&S_addr, S_block + BLOCK_SIZE - sizeof(int), sizeof(int));
            freeBlockInBuffer(S_block, buf);
        }

        // 获取下一个 R 块
        memcpy(&R_addr, R_block + BLOCK_SIZE - sizeof(int), sizeof(int));
        freeBlockInBuffer(R_block, buf);
    }

    // 写入最后一个结果块
    if (result_block != NULL) {
        int end_addr = 0;
        memcpy(result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));
        writeBlockToDisk(result_block, result_addr, buf);
        freeBlockInBuffer(result_block, buf);
    }

    printf("+------+------+------+------+\n");
    printf("Join results written to disk starting from block %d\n", result_start_addr);
    printf("Total Join tuple numbles: %d\n",total_join_num);
}

// 主函数
int main() {
    Buffer buf;
    if (!initBuffer(BUFFER_SIZE, BLOCK_SIZE, &buf)) {
        printf("Buffer initialization failed!\n");
        return -1;
    }


    Relation R = {1, 2, {0, 4}, {"A", "B"}}; // R 的起始地址为 1，属性为 A 和 B
    Relation S = {100, 2, {0, 4}, {"C", "D"}}; // S 的起始地址为 17，属性为 C 和 D

    // 执行连接
    unsigned int result_start_addr = 220;  // 假设结果存储从块 100 开始
    nested_loop_join(&buf, &R, &S, result_start_addr);
    
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

// gcc -o  nested_loop_join nested_loop_join.c extmem.c
// ./nested_loop_join 
// od -t dI -An 314.blk
// od -t dI -An 315.blk