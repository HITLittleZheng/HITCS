#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extmem.h"

#define R_BLOCK_COUNT 16
#define S_BLOCK_COUNT 32
#define TUPLE_SIZE 8
#define BLOCK_SIZE 64
#define BUFFER_SIZE 520

// 查看数据 ： od -t dI -An 1.blk
// 打印块内容，便于调试和可视化
void print_block_content(unsigned char *block, int block_id) {
    printf("Block %d:\n", block_id);
    unsigned char *ptr = block;

    // 遍历并打印元组
    for (int i = 0; i < (BLOCK_SIZE - sizeof(int)) / TUPLE_SIZE; i++) {
        int attr1, attr2;
        memcpy(&attr1, ptr, sizeof(int));
        ptr += sizeof(int);
        memcpy(&attr2, ptr, sizeof(int));
        ptr += sizeof(int);
        printf("  Tuple (%d, %d)\n", attr1, attr2);
    }

    // 打印下一个块的地址
    int next_block;
    memcpy(&next_block, block + BLOCK_SIZE - sizeof(int), sizeof(int));
    if (next_block == 0) {
        printf("  Next block: NULL (end of chain)\n");
    } else {
        printf("  Next block: %d\n", next_block);
    }
}

// 生成关系 R 或 S
void generate_relation(Buffer *buf, unsigned int start_addr, int block_count, int value_a_min, int value_a_max, int value_b_min, int value_b_max) {
    unsigned int addr = start_addr;

    for (int i = 0; i < block_count; i++) {
        // 获取新块
        unsigned char *block = getNewBlockInBuffer(buf);
        if (!block) {
            printf("Failed to allocate block\n");
            return;
        }

        unsigned char *ptr = block;

        // 在块内写入元组
        for (int j = 0; j < (BLOCK_SIZE - sizeof(int)) / TUPLE_SIZE; j++) {
            int attr1 = value_a_min + rand() % (value_a_max - value_a_min + 1);
            int attr2 = value_b_min + rand() % (value_b_max - value_b_min + 1);

            memcpy(ptr, &attr1, sizeof(int));
            ptr += sizeof(int);
            memcpy(ptr, &attr2, sizeof(int));
            ptr += sizeof(int);
        }

        // 写入下一个块的地址到最后 4 个字节
        if (i < block_count - 1) {
            int next_addr = addr + 1;
            memcpy(block + BLOCK_SIZE - sizeof(int), &next_addr, sizeof(int));
        } else {
            int end_addr = 0; // 表示链表结束
            memcpy(block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));
        }

        // 打印当前块内容
        printf("Generated content for block %d:\n", addr);
        print_block_content(block, addr);

        // 写入磁盘
        if (writeBlockToDisk(block, addr, buf) != 0) {
            printf("Failed to write block %d\n", addr);
            return;
        }
        addr++;
    }

    printf("Generated %d blocks starting from address %d\n", block_count, start_addr);
}

int main() {
    Buffer buf;
    if (!initBuffer(BUFFER_SIZE, BLOCK_SIZE, &buf)) {
        printf("Buffer initialization failed!\n");
        return -1;
    }

    // 生成关系 R
    printf("Generating relation R...\n");
    generate_relation(&buf, 1, R_BLOCK_COUNT, 1, 40, 1, 1000);

    // 生成关系 S
    printf("Generating relation S...\n");
    generate_relation(&buf, 100, S_BLOCK_COUNT, 20, 60, 1, 1000);

    freeBuffer(&buf);
    return 0;
}

// gcc -o  data_gen data_gen.c extmem.c