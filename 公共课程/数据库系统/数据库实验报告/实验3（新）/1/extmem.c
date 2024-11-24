/*
 * extmem.c
 * Zhaonian Zou
 * Harbin Institute of Technology
 * Jun 22, 2011
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "extmem.h"

Buffer *initBuffer(size_t bufSize, size_t blkSize, Buffer *buf)
{
    int i;

    buf->numIO = 0;
    buf->bufSize = bufSize;
    buf->blkSize = blkSize;
    buf->numAllBlk = bufSize / (blkSize + 1);
    buf->numFreeBlk = buf->numAllBlk;
    buf->data = (unsigned char *)malloc(bufSize * sizeof(unsigned char));

    if (!buf->data)
    {
        perror("Buffer Initialization Failed!\n");
        return NULL;
    }

    memset(buf->data, 0, bufSize * sizeof(unsigned char));
    return buf;
}

void freeBuffer(Buffer *buf)
{
    free(buf->data);
}

unsigned char *getNewBlockInBuffer(Buffer *buf)
{
    unsigned char *blkPtr;

    if (buf->numFreeBlk == 0)
    {
        perror("Buffer is full!\n");
        return NULL;
    }

    blkPtr = buf->data;

    while (blkPtr < buf->data + (buf->blkSize + 1) * buf->numAllBlk)
    {
        if (*blkPtr == BLOCK_AVAILABLE)
            break;
        else
            blkPtr += buf->blkSize + 1;
    }

    *blkPtr = BLOCK_UNAVAILABLE;
    buf->numFreeBlk--;
    return blkPtr + 1;
}

void freeBlockInBuffer(unsigned char *blk, Buffer *buf)
{
    *(blk - 1) = BLOCK_AVAILABLE;
    buf->numFreeBlk++;
}

int dropBlockOnDisk(unsigned int addr)
{
    char filename[40];

    sprintf(filename, "%d.blk", addr);

    if (remove(filename) == -1)
    {
        perror("Dropping Block Fails!\n");
        return -1;
    }

    return 0;
}

unsigned char *readBlockFromDisk(unsigned int addr, Buffer *buf)
{
    char filename[40];
    unsigned char *blkPtr, *bytePtr;
    char ch;

    if (buf->numFreeBlk == 0)
    {
        perror("Buffer Overflows!\n");
        return NULL;
    }

    blkPtr = buf->data;

    while (blkPtr < buf->data + (buf->blkSize + 1) * buf->numAllBlk)
    {
        if (*blkPtr == BLOCK_AVAILABLE)
            break;
        else
            blkPtr += buf->blkSize + 1;
    }

    sprintf(filename, "%d.blk", addr);
    FILE *fp = fopen(filename, "r");

    if (!fp)
    {
        perror("Reading Block Failed!\n");
        return NULL;
    }

    *blkPtr = BLOCK_UNAVAILABLE;
    blkPtr++;
    bytePtr = blkPtr;

    while ((ch = fgetc(fp)) != EOF && bytePtr < blkPtr + buf->blkSize)
    {
        *bytePtr = ch;
        bytePtr++;
    }

    fclose(fp);
    buf->numFreeBlk--;
    buf->numIO++;
    return blkPtr;
}

int writeBlockToDisk(unsigned char *blkPtr, unsigned int addr, Buffer *buf)
{
    char filename[40];
    unsigned char *bytePtr;

    sprintf(filename, "%d.blk", addr);
    FILE *fp = fopen(filename, "w");

    if (!fp)
    {
        perror("Writing Block Failed!\n");
        return -1;
    }

    for (bytePtr = blkPtr; bytePtr < blkPtr + buf->blkSize; bytePtr++)
        fputc((int)(*bytePtr), fp);

    fclose(fp);
    *(blkPtr - 1) = BLOCK_AVAILABLE;
    buf->numFreeBlk++;
    buf->numIO++;
    return 0;
}
