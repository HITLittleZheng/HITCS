#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include "extmem.h"  

#define BLOCK_SIZE 64  
#define BUFFER_SIZE 520  
#define MAX_TUPLES_PER_BLOCK ((BLOCK_SIZE - sizeof(int)) / 8)  // 每个块最多存储的元组数量  
#define PARTITION_COUNT 5   // 分区数（M-1）  

// 定义关系模式结构体  
typedef struct {  
    unsigned int start_addr; // 起始磁盘块地址  
    int attr_count;          // 属性个数  
    int tuple_size;  
    int attr_offsets[2];     // 属性偏移量（仅支持 2 个属性 A/B 或 C/D）  
    char attr_names[2][10];  // 属性名称  
    char rel_name[10];  
} Relation;  

// 哈希表节点  
typedef struct BucketNode {  
    int key;                // 存储哈希值，连接的左侧属性值  
    unsigned char *result_block;   
    int result_count;      
    unsigned int addr;  
    unsigned int result_addr;      
} BucketNode;  

// 哈希函数：简单的模哈希  
unsigned int bucket(int key) {  
    return key % PARTITION_COUNT;  
}  

// 哈希分区  
void partition(Buffer *buf, Relation *rel, int attr_index, BucketNode bucketTable[PARTITION_COUNT]) {  
    unsigned char *block;  
    unsigned int current_addr = rel->start_addr;  

    // 将 R 分区  
    printf("Partitioning relation ...\n");  
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
            int bucket_value;  
            memcpy(&attr_value, ptr + rel->attr_offsets[attr_index], sizeof(int)); // 获取指定属性值  
            bucket_value = bucket(attr_value);  

            // 如果结果块为空或已满，分配新块  
            if (bucketTable[bucket_value].result_block == NULL || bucketTable[bucket_value].result_count >= MAX_TUPLES_PER_BLOCK) {  
                if (bucketTable[bucket_value].result_block != NULL) {  
                    // 写入结果块的后继地址  
                    int next_addr = bucketTable[bucket_value].result_addr + 1;  
                    memcpy(bucketTable[bucket_value].result_block + BLOCK_SIZE - sizeof(int), &next_addr, sizeof(int));  
                    writeBlockToDisk(bucketTable[bucket_value].result_block, bucketTable[bucket_value].result_addr, buf);  
                    freeBlockInBuffer(bucketTable[bucket_value].result_block, buf);  
                    bucketTable[bucket_value].result_addr++;  
                }  

                // 分配新结果块  
                bucketTable[bucket_value].result_block = getNewBlockInBuffer(buf);  
                if (!bucketTable[bucket_value].result_block) { 
                    printf("Buffer full while allocating result block!\n");  
                    return;  
                }  
                memset(bucketTable[bucket_value].result_block, 0, BLOCK_SIZE);  
                bucketTable[bucket_value].result_count = 0;  
            }  

            // 将满足条件的元组写入结果块  
            memcpy(bucketTable[bucket_value].result_block + bucketTable[bucket_value].result_count * 8, ptr, 8);  
            bucketTable[bucket_value].result_count++; 

            ptr += 8; // 移动到下一个元组  
        }  

        // 获取当前块的后继地址  
        memcpy(&current_addr, block + BLOCK_SIZE - sizeof(int), sizeof(int));  
        freeBlockInBuffer(block, buf);  
    }  
    // 写入最后一个结果块  
    for(int i = 0; i < PARTITION_COUNT; i++) {  
        if (bucketTable[i].result_count > 0) {  
            int end_addr = 0;  
            memcpy(bucketTable[i].result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));  
            writeBlockToDisk(bucketTable[i].result_block, bucketTable[i].result_addr, buf);  
            freeBlockInBuffer(bucketTable[i].result_block, buf);  
        }  
    }  
}  

// 在一个桶内执行连接操作
void joinPartitions(Buffer *buf, BucketNode bucketTable_R[PARTITION_COUNT], BucketNode bucketTable_S[PARTITION_COUNT], unsigned int result_base_addr) {
    unsigned char *R_block, *S_block, *result_block = NULL;
    int result_count = 0;
    unsigned int result_addr = result_base_addr;
    int total_join_num = 0;

    printf("Performing Hash Join (R.A = S.C)\n");
    printf("+------+------+------+------+\n");
    printf("|  R.A |  R.B |  S.C |  S.D |\n");
    printf("+------+------+------+------+\n");

    for (int i = 0; i < PARTITION_COUNT; i++) {
        printf("Joining partition %d...\n", i);

        unsigned int R_addr = bucketTable_R[i].addr;
        unsigned int S_addr = bucketTable_S[i].addr;

        while (R_addr != 0) {
            R_block = readBlockFromDisk(R_addr, buf);
            if (!R_block) {
                printf("Failed to read block %d for R partition %d\n", R_addr, i);
                return;
            }

            unsigned char *R_ptr = R_block;
            for (int j = 0; j < MAX_TUPLES_PER_BLOCK; j++) {
                int R_A, R_B;
                memcpy(&R_A, R_ptr, sizeof(int));       // 读取 R.A
                memcpy(&R_B, R_ptr + 4, sizeof(int));   // 读取 R.B

                unsigned int current_S_addr = S_addr;
                while (current_S_addr != 0) {
                    S_block = readBlockFromDisk(current_S_addr, buf);
                    if (!S_block) {
                        printf("Failed to read block %d for S partition %d\n", current_S_addr, i);
                        return;
                    }

                    unsigned char *S_ptr = S_block;
                    for (int k = 0; k < MAX_TUPLES_PER_BLOCK; k++) {
                        int S_C, S_D;
                        memcpy(&S_C, S_ptr, sizeof(int));       // 读取 S.C
                        memcpy(&S_D, S_ptr + 4, sizeof(int));   // 读取 S.D

                        // 检查连接条件
                        if (R_A == S_C && R_A != 0) {
                            // 如果结果块为空或已满，分配新块
                            if (result_block == NULL || result_count == (BLOCK_SIZE - sizeof(int)) / 16) {
                                if (result_block != NULL) {
                                    int next_addr = result_addr + 1;
                                    memcpy(result_block + BLOCK_SIZE - sizeof(int), &next_addr, sizeof(int));
                                    writeBlockToDisk(result_block, result_addr, buf);
                                    freeBlockInBuffer(result_block, buf);
                                    result_addr++;
                                }

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
                            memcpy(result_block + result_count * 16 + 4, &R_B, sizeof(int));  // R.B
                            memcpy(result_block + result_count * 16 + 8, &S_C, sizeof(int));  // S.C
                            memcpy(result_block + result_count * 16 + 12, &S_D, sizeof(int)); // S.D
                            result_count++;
                            total_join_num++;

                            // 打印连接结果
                            printf("| %-4d | %-4d | %-4d | %-4d |\n", R_A, R_B, S_C, S_D);
                        }

                        S_ptr += 8; // 移动到下一个元组
                    }

                    memcpy(&current_S_addr, S_block + BLOCK_SIZE - sizeof(int), sizeof(int));
                    freeBlockInBuffer(S_block, buf);
                }

                R_ptr += 8; // 移动到下一个元组
            }

            memcpy(&R_addr, R_block + BLOCK_SIZE - sizeof(int), sizeof(int));
            freeBlockInBuffer(R_block, buf);
        }
    }

    // 写入最后一个结果块
    if (result_block != NULL) {
        int end_addr = 0;
        memcpy(result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));
        writeBlockToDisk(result_block, result_addr, buf);
        freeBlockInBuffer(result_block, buf);
    }

    printf("+------+------+------+------+\n");
    printf("Join results written to disk starting from block %d\n", result_base_addr);
    printf("Total Join tuple numbles: %d\n",total_join_num);
}

// 主函数
int main() {
    Buffer buf;
    if (!initBuffer(BUFFER_SIZE, BLOCK_SIZE, &buf)) {
        printf("Buffer initialization failed!\n");
        return -1;
    }

    Relation R = {1, 2, 8, {0, 4}, {"A", "B"}, "R"}; // R 的起始地址为 1，属性为 A 和 B
    Relation S = {100, 2, 8, {0, 4}, {"C", "D"}, "S"}; // S 的起始地址为 100，属性为 C 和 D

    BucketNode bucketTable_R[PARTITION_COUNT] = {0};
    BucketNode bucketTable_S[PARTITION_COUNT] = {0};

    unsigned int base_addr_R = 300;
    unsigned int base_addr_S = 360;
    unsigned int result_base_addr = 420;

    for (int i = 0; i < PARTITION_COUNT; i++) {
        bucketTable_R[i].addr = base_addr_R + i * 10;
        bucketTable_S[i].addr = base_addr_S + i * 10;
        bucketTable_R[i].result_addr = base_addr_R + i * 10;
        bucketTable_S[i].result_addr = base_addr_S + i * 10;
        bucketTable_R[i].result_block = NULL;
        bucketTable_R[i].result_count = 0;
        bucketTable_S[i].result_block = NULL;
        bucketTable_S[i].result_count = 0;
    }

    partition(&buf, &R, 0, bucketTable_R);
    partition(&buf, &S, 0, bucketTable_S);

    joinPartitions(&buf, bucketTable_R, bucketTable_S, result_base_addr);

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

// gcc -o  hash-join Hash-Join.c extmem.c
// od -t dI -An 352.blk
// od -t dI -An 470.blk