#ifndef WM_H
#define WM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASHTABLESIZE (256*256)
#define MAXLEN 256

typedef struct wm_pattern_struct//每个模式串的结构
{
	struct wm_pattern_struct *next;//指向下一个模式串
	unsigned char *psPat; //pattern array//模式串数组
	unsigned psLen; //length of pattern in bytes//模式串的长度
}WM_PATTERN_STRUCT;

#define HASH_TYPE short
#define SHIFTTABLESIZE (256*256)

typedef struct wm_struct//模式串集的结构
{
	WM_PATTERN_STRUCT *plist; //pattern list模式子串列表
	WM_PATTERN_STRUCT *msPatArray; //array of patterns模式子串数组（队列）
	unsigned short *msNumArray; //array of group counts, # of patterns in each hash group
	int msNumPatterns; //number of patterns loaded//模式子串的个数
	unsigned msNumHashEntries;//HASH表的大小
	HASH_TYPE *msHash; //last 2 characters pattern hash table//HASH表
	unsigned char* msShift; //bad word shift table//SHIFT表
	HASH_TYPE *msPrefix; //first 2 characters prefix table//PREFIX表
	int msSmallest; //shortest length of all patterns//最短模式子串的长度
}WM_STRUCT;

//函数声明
WM_STRUCT * wmNew();  //创建模式串集函数
void wmFree(WM_STRUCT *ps); //释放空间函数
int wmAddPattern(WM_STRUCT *ps,unsigned char *P,int m); //添加子模式串函数
int wmPrepPatterns(WM_STRUCT *ps); //预处理函数
void wmSearch(WM_STRUCT *ps,unsigned char *Tx,int n); //模式匹配函数

#endif