#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <limits.h>   
#include "extmem.h"  

#define BLOCK_SIZE 64  
#define BUFFER_SIZE 780  
#define MAX_TUPLES_PER_BLOCK ((BLOCK_SIZE - sizeof(int)) / 8) // 每块最多元组数  
#define MAX_SEGMENT 10
#define TUPLE_SIZE 8

typedef struct {  
    unsigned int start_addr; // 起始磁盘块地址  
    int attr_count;          // 属性个数  
    int tuple_size;          // 元组大小  
    int attr_offsets[2];     // 属性偏移量（仅支持2个属性）  
    char attr_names[2][10];  // 属性名称  
    char rel_name[10];       // 关系名称  
} Relation;  

// 比较函数：用于 qsort，控制元组放在最后，按第一个属性升序排序  
int compareTuples(const void *a, const void *b) {  
    const int *tuple_a = (const int *)a;  
    const int *tuple_b = (const int *)b;  

    // 如果 a 或 b 是控制元组（任一属性值为 0），放在最后  
    if (tuple_a[0] == 0 || tuple_a[1] == 0) return 1; // a 是控制元组  
    if (tuple_b[0] == 0 || tuple_b[1] == 0) return -1; // b 是控制元组  

    // 否则按第一个属性值排序  
    return tuple_a[0] - tuple_b[0];  
}  

// 对关系进行排序并划分为多个段  
int sortRelation(Buffer *buf, Relation *rel, unsigned int segs[]) {  
    unsigned int current_addr = rel->start_addr, result_addr;  
    unsigned char *block;  
    int seg_count = 0;                          // 当前段计数  
    int block_count = 0;                        // 当前段的块计数  
    unsigned char *segment_buffer[MAX_SEGMENT] = {NULL}; // 段缓冲区  
    unsigned char *segment_ptr[MAX_SEGMENT] = {NULL};    // 指向段缓冲区的指针  
    unsigned char *result_block = NULL;         // 结果块  
    int result_count = 0;                       // 结果元组计数 

    printf("Sorting relation %s...\n", rel->rel_name);  

    // 处理每个块  
    while (current_addr != 0) {  
        block = readBlockFromDisk(current_addr, buf);  
        if (!block) {  
            printf("Failed to read block %d\n", current_addr);  
            return -1;  
        }  

        // 排序当前块  
        qsort(block, MAX_TUPLES_PER_BLOCK, rel->tuple_size, compareTuples);  

        // 将块存入段缓冲区  
        if (block_count < MAX_SEGMENT) { // 确保不越界  
            segment_ptr[block_count] = block;  
            segment_buffer[block_count] = block;  
            block_count++;  
        } else {  
            printf("Block count exceeded MAX_SEGMENT\n");  
            freeBlockInBuffer(block, buf); // 清理内存  
            break; // 退出循环  
        }  

        // 获取下一个块地址  
        memcpy(&current_addr, block + BLOCK_SIZE - sizeof(int), sizeof(int));  

        // 如果到达归并段的块个数或到达最后一个块  
        if (block_count == MAX_SEGMENT || current_addr == 0) {  
            // 初始化这个归并段的起始地址  
            result_addr = segs[seg_count];  
            while(1) {  
                // 找到几个段中最小的元组  
                int min_seg = -1;  
                int min_value = INT_MAX;  

                for (int i = 0; i < block_count; i++) {  
                    if (segment_ptr[i]) {  
                        int value;  
                        memcpy(&value, segment_ptr[i], sizeof(int));  
                        if (value < min_value && value != 0) {  
                            min_value = value;  
                            min_seg = i;  
                        }  
                    }  
                }  

                // 如果没有找到有效值，归并结束  
                if (min_seg == -1) break;  

                // 如果缓冲区满则写入缓存并添加新的缓冲区  
                if (result_block == NULL || result_count >= (BLOCK_SIZE - sizeof(int)) / 8) {  
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
                        return -1; // 修正返回值  
                    }  

                    memset(result_block, 0, BLOCK_SIZE);  
                    result_count = 0;  
                }  

                // 把最小的元组写入缓冲区    
                memcpy(result_block + result_count * rel->tuple_size, segment_ptr[min_seg], rel->tuple_size);  
                result_count++;  
                segment_ptr[min_seg] += rel->tuple_size; // 假设每个元组大小是rel->tuple_size  
                if (segment_ptr[min_seg] >= segment_buffer[min_seg] + BLOCK_SIZE - sizeof(int)) {  
                    segment_ptr[min_seg] = NULL;  
                }  
            }  

            // 写最后一个结果块  
            if (result_count > 0) {  
                int end_addr = 0;  
                memcpy(result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));  
                writeBlockToDisk(result_block, result_addr, buf);  
                freeBlockInBuffer(result_block, buf);  
                result_block = NULL; // 清空指针方便后续重用与检测  
            }  

            // 清理段缓冲区  
            for (int k = 0; k < block_count; k++) {  
                freeBlockInBuffer(segment_buffer[k], buf);  
                segment_buffer[k] = NULL;  
                segment_ptr[k] = NULL;  
            }  

            printf("Segment %d Merging completed.\n", seg_count);  
            seg_count++; // 增加段计数  
            block_count = 0; // 重置块计数以处理下一个段  
        }  
    }  

    return seg_count; // 返回段计数  
}  


// 从块中获取元组
void getTuple(unsigned char *block, int tuple_index, int *attr1, int *attr2) {
    memcpy(attr1, block + tuple_index * TUPLE_SIZE, sizeof(int));
    memcpy(attr2, block + tuple_index * TUPLE_SIZE + 4, sizeof(int));
}


void multiwayMergeJoin(Buffer *buf, unsigned int *R_runs, unsigned int R_count, unsigned int *S_runs, unsigned int S_count, unsigned int result_start_addr) {
    unsigned char *R_blocks[R_count];
    unsigned char *S_blocks[S_count];
    unsigned int R_addrs[R_count], S_addrs[S_count];
    unsigned int result_addr = result_start_addr;
    unsigned char *result_block = getNewBlockInBuffer(buf);
    int result_count = 0;
    int total_join_num = 0; 
    
    printf("Performing Hash Join (R.A = S.C)\n");
    printf("+------+------+------+------+\n");
    printf("|  R.A |  R.B |  S.C |  S.D |\n");
    printf("+------+------+------+------+\n");


    // 初始化各归并段的地址和读取的块
    for (int i = 0; i < R_count; i++) {
        R_addrs[i] = R_runs[i];
        R_blocks[i] = readBlockFromDisk(R_addrs[i], buf);
    }
    for (int i = 0; i < S_count; i++) {
        S_addrs[i] = S_runs[i];
        S_blocks[i] = readBlockFromDisk(S_addrs[i], buf);
    }

    int R_values[R_count][2]; // 存储每个段当前元组
    int S_values[S_count][2];
    int R_indices[R_count], S_indices[S_count]; // 每段当前元组的索引

    for (int i = 0; i < R_count; i++) R_indices[i] = 0;
    for (int i = 0; i < S_count; i++) S_indices[i] = 0;

    // 初始加载每段的第一个元组
    for (int i = 0; i < R_count; i++) {
        if (R_blocks[i]) getTuple(R_blocks[i], R_indices[i], &R_values[i][0], &R_values[i][1]);
    }
    for (int i = 0; i < S_count; i++) {
        if (S_blocks[i]) getTuple(S_blocks[i], S_indices[i], &S_values[i][0], &S_values[i][1]);
    }

    int S_buffer[MAX_TUPLES_PER_BLOCK * S_count][2]; // 缓存当前最小值的所有 S 元组
    int S_buffer_size = 0;

    // 开始归并连接
    while (1) {
        int min_R = INT_MAX, min_S = INT_MAX;

        // 找到 R 和 S 的当前最小值
        for (int i = 0; i < R_count; i++) {
            if (R_blocks[i] && R_indices[i] < MAX_TUPLES_PER_BLOCK) {
                if (R_values[i][0] < min_R) {
                    min_R = R_values[i][0];
                }
            }
        }
        for (int i = 0; i < S_count; i++) {
            if (S_blocks[i] && S_indices[i] < MAX_TUPLES_PER_BLOCK) {
                if (S_values[i][0] < min_S) {
                    min_S = S_values[i][0];
                }
            }
        }

        // 如果都没有可用元组，结束
        if (min_R == INT_MAX || min_S == INT_MAX) break;

        // 如果 R 和 S 的最小值相等，处理匹配
        if (min_R == min_S) {
            S_buffer_size = 0;

            // 缓存所有等于 min_S 的 S 元组
            for (int i = 0; i < S_count; i++) {
                while (S_blocks[i] && S_indices[i] < MAX_TUPLES_PER_BLOCK && S_values[i][0] == min_S) {
                    S_buffer[S_buffer_size][0] = S_values[i][0];
                    S_buffer[S_buffer_size][1] = S_values[i][1];
                    S_buffer_size++;

                    // 移动 S 的指针
                    S_indices[i]++;
                    if (S_indices[i] == MAX_TUPLES_PER_BLOCK) {
                        unsigned int next_addr;
                        memcpy(&next_addr, S_blocks[i] + BLOCK_SIZE - sizeof(int), sizeof(int));
                        freeBlockInBuffer(S_blocks[i], buf);
                        if (next_addr != 0) {
                            S_blocks[i] = readBlockFromDisk(next_addr, buf);
                            S_indices[i] = 0;
                            getTuple(S_blocks[i], 0, &S_values[i][0], &S_values[i][1]);
                        } else {
                            S_blocks[i] = NULL;
                        }
                    } else {
                        getTuple(S_blocks[i], S_indices[i], &S_values[i][0], &S_values[i][1]);
                    }
                }
            }

            // 遍历所有等于 min_R 的 R 元组，与 S_buffer 中的元组连接
            for (int i = 0; i < R_count; i++) {
                while (R_blocks[i] && R_indices[i] < MAX_TUPLES_PER_BLOCK && R_values[i][0] == min_R) {
                    for (int j = 0; j < S_buffer_size; j++) {
                        // 输出匹配的结果
                        if (result_count == (BLOCK_SIZE - sizeof(int)) / (2 * TUPLE_SIZE)) {
                            int next_addr = result_addr + 1;
                            memcpy(result_block + BLOCK_SIZE - sizeof(int), &next_addr, sizeof(int));
                            writeBlockToDisk(result_block, result_addr, buf);
                            freeBlockInBuffer(result_block, buf);
                            result_block = getNewBlockInBuffer(buf);
                            result_count = 0;
                        }

                        memcpy(result_block + result_count * 16, &R_values[i][0], sizeof(int));
                        memcpy(result_block + result_count * 16 + 4, &R_values[i][1], sizeof(int));
                        memcpy(result_block + result_count * 16 + 8, &S_buffer[j][0], sizeof(int));
                        memcpy(result_block + result_count * 16 + 12, &S_buffer[j][1], sizeof(int));
                        // 打印连接结果
                        printf("| %-4d | %-4d | %-4d | %-4d |\n", R_values[i][0], R_values[i][1], S_buffer[j][0], S_buffer[j][1]);
                        result_count++;
                        total_join_num++;
                    }

                    // 移动 R 的指针
                    R_indices[i]++;
                    if (R_indices[i] == MAX_TUPLES_PER_BLOCK) {
                        unsigned int next_addr;
                        memcpy(&next_addr, R_blocks[i] + BLOCK_SIZE - sizeof(int), sizeof(int));
                        freeBlockInBuffer(R_blocks[i], buf);
                        if (next_addr != 0) {
                            R_blocks[i] = readBlockFromDisk(next_addr, buf);
                            R_indices[i] = 0;
                            getTuple(R_blocks[i], 0, &R_values[i][0], &R_values[i][1]);
                        } else {
                            R_blocks[i] = NULL;
                        }
                    } else {
                        getTuple(R_blocks[i], R_indices[i], &R_values[i][0], &R_values[i][1]);
                    }
                }
            }
        } else {
            // 如果 min_R 和 min_S 不相等，移动较小值的指针
            if (min_R < min_S) {
                for (int i = 0; i < R_count; i++) {
                    if (R_blocks[i] && R_indices[i] < MAX_TUPLES_PER_BLOCK && R_values[i][0] == min_R) {
                        R_indices[i]++;
                        if (R_indices[i] == MAX_TUPLES_PER_BLOCK) {
                            unsigned int next_addr;
                            memcpy(&next_addr, R_blocks[i] + BLOCK_SIZE - sizeof(int), sizeof(int));
                            freeBlockInBuffer(R_blocks[i], buf);
                            if (next_addr != 0) {
                                R_blocks[i] = readBlockFromDisk(next_addr, buf);
                                R_indices[i] = 0;
                                getTuple(R_blocks[i], 0, &R_values[i][0], &R_values[i][1]);
                            } else {
                                R_blocks[i] = NULL;
                            }
                        } else {
                            getTuple(R_blocks[i], R_indices[i], &R_values[i][0], &R_values[i][1]);
                        }
                    }
                }
            } else {
                for (int i = 0; i < S_count; i++) {
                    if (S_blocks[i] && S_indices[i] < MAX_TUPLES_PER_BLOCK && S_values[i][0] == min_S) {
                        S_indices[i]++;
                        if (S_indices[i] == MAX_TUPLES_PER_BLOCK) {
                            unsigned int next_addr;
                            memcpy(&next_addr, S_blocks[i] + BLOCK_SIZE - sizeof(int), sizeof(int));
                            freeBlockInBuffer(S_blocks[i], buf);
                            if (next_addr != 0) {
                                S_blocks[i] = readBlockFromDisk(next_addr, buf);
                                S_indices[i] = 0;
                                getTuple(S_blocks[i], 0, &S_values[i][0], &S_values[i][1]);
                            } else {
                                S_blocks[i] = NULL;
                            }
                        } else {
                            getTuple(S_blocks[i], S_indices[i], &S_values[i][0], &S_values[i][1]);
                        }
                    }
                }
            }
        }
    }

    // 写入最后的结果块
    if (result_count > 0) {
        int end_addr = 0;
        memcpy(result_block + BLOCK_SIZE - sizeof(int), &end_addr, sizeof(int));
        writeBlockToDisk(result_block, result_addr, buf);
        freeBlockInBuffer(result_block, buf);
    }
    printf("+------+------+------+------+\n");
    printf("Join completed. Results start at block %d\n", result_start_addr);
    printf("Total Join tuple numbles: %d\n",total_join_num);
}


// 主函数  
int main() {  
    Buffer buf;  
    if (!initBuffer(BUFFER_SIZE, BLOCK_SIZE, &buf)) {  
        printf("Buffer initialization failed!\n");  
        return -1;  
    }  
    unsigned int result_base_addr = 800;

    Relation R = {1, 2, 8, {0, 4}, {"A", "B"}, "R"};  
    Relation S = {100, 2, 8, {0, 4}, {"C", "D"}, "S"}; 

    unsigned int segs_R[10];  
    for (int i = 0; i < 10; i++) {   
        segs_R[i] = 600 + 10 * i;   
    }  

    unsigned int segs_S[10];  
    for (int i = 0; i < 10; i++) {   
        segs_S[i] = 700 + 10 * i;   
    }  

    // Sort the relation and get the number of segments merged  
    int seg_count_R = sortRelation(&buf, &R, segs_R); 
    int seg_count_S = sortRelation(&buf, &S, segs_S);  
 
    printf("R Successfully merged %d segments.\n", seg_count_R);  
    printf("S Successfully merged %d segments.\n", seg_count_S);  
    
    multiwayMergeJoin(&buf, segs_R, seg_count_R, segs_S, seg_count_S, result_base_addr);

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

// gcc -o sort-merge-join sort-merge-join.c extmem.c
// ./sort-merge-join
// od -t dI -An 600.blk
// od -t dI -An 601.blk
// od -t dI -An 602.blk
// od -t dI -An 603.blk
// od -t dI -An 604.blk
