#include "wm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

extern int nline;
extern int nfound;
#define MAXN 10001 //模式串的最大长度MAXN - 1
#define MAXM 51//单词最大长度为MAXM - 1

/* ****************************************************************
    函数：void wmNew()
	目的：创建一个模式串集合
	参数：
		无
    返回：
	    WM_STRUCT - 创建的模式串集
 ****************************************************************/
WM_STRUCT * wmNew()
{
	WM_STRUCT *p=(WM_STRUCT *)malloc(sizeof(WM_STRUCT));
	if(!p) return 0;
	p->msNumPatterns=0;//模式串的个数,初始为0
	p->msSmallest=1000;//最短模式串的长度
	return p;
}

/* ****************************************************************
    函数：void wmFree(WM_STRUCT *)
	目的：释放模式串集占用空间
	参数：
		ps => 模式串集
    返回：
	    
 ****************************************************************/
void wmFree(WM_STRUCT *ps) //释放空间函数
{
	if(ps->msPatArray) //如果模式串集中存在子串，则先释放子串数组占用空间
	{
		if(ps->msPatArray->psPat) free(ps->msPatArray->psPat);	//子串不为空，则释放
		free(ps->msPatArray );
	}
	if(ps->msNumArray) free(ps->msNumArray);
	if(ps->msHash) free(ps->msHash);
	if(ps->msPrefix) free(ps->msPrefix);
	if(ps->msShift) free(ps->msShift);
	free(ps);
}

/* ****************************************************************
    函数：int wmAddPattern(WM_STRUCT *,unsigned char *,int )
	目的：向模式串集ps中新增一个长度为m的子串q
	参数：
		ps => 模式串集
		q => 要新增的子串
		m => 子串长度
    返回：
	    int* - 新增成功0，失败-1
 ****************************************************************/
int wmAddPattern(WM_STRUCT *ps,unsigned char *q,int m)
{
	WM_PATTERN_STRUCT *p;  //定义一个子串结构
	p=(WM_PATTERN_STRUCT *)malloc(sizeof(WM_PATTERN_STRUCT));
	if(!p) return -1;

	p->psPat=(unsigned char*)malloc(m+1); //据子串数组的长度分配空间
	memset(p->psPat+m,0,1);	//最后一个位置设置为结束字符“/0” 
	memcpy(p->psPat,q,m); //拷贝q到子串结构数组中
	p->psLen=m; //子串长度赋值
	ps->msNumPatterns++; //模式串集的子串个数增1
	if(p->psLen < (unsigned)ps->msSmallest) ps->msSmallest = p->psLen; //重新确定最短字符串长度

	p->next=ps->plist; //将新增子串加入字符串集列表中。队列形式，新增在队列头部
	ps->plist=p;

	return 0;
}

/* ****************************************************************
    函数：static unsigned HASH16(unsigned char *)
	目的：对一串字符进行哈希计算。计算方式为：(((*T)<<8) | *(T+1))，
	参数：
		T => 要哈希计算的字符串
    返回：
	    unsigned - 静态函数，返回对字符串T计算的哈希值
 ****************************************************************/
static unsigned HASH16(unsigned char *T)
{
	/*/
	printf("T:%c\n",*(T));
	getchar();
	printf("T+1:%c\n",*(T+1));
	getchar();
	printf("T<<8:%c\n",(int)((*T)<<8));
	getchar();
	printf("HASH16:%d\n",((*T)<<8) | *(T+1));
	getchar();
	//*/
	return (unsigned short) (((*T)<<8) | *(T+1)); //对第一个字符左移8位，然后与第二个字符异或运算
}

/* ****************************************************************
    函数：sort(WM_STRUCT *)
	目的：对字符串集ps中的子串队列，根据子串串值的哈希值从小到大排序
	参数：
		ps => 模式串集
    返回：无
 ****************************************************************/
void sort(WM_STRUCT *ps)
{
	int m=ps->msSmallest; //获取最短子串长度
	int i,j;
	unsigned char *temp;
	int flag;	//冒泡排序的标志位。当一趟比较无交换时，说明已经完成排序，即可以跳出循环结束
	for(i = ps->msNumPatterns-1,flag = 1;i > 0 && flag;i--)  //循环对字符串集中的每个子串，根据其哈希值大小进行冒泡排序
	{
		flag=0;
		for(j=0;j<i;j++)
		{
			if(HASH16(&(ps->msPatArray[j+1].psPat[m-2]))<HASH16(&(ps->msPatArray[j].psPat[m-2])))//比较的为每个子串截取部分的最后两个字符的哈希值
			{
				flag=1;
				temp=ps->msPatArray[j+1].psPat;
				ps->msPatArray[j+1].psPat=ps->msPatArray[j].psPat;
				ps->msPatArray[j].psPat=temp;
			}
		}
	}
}

/* ****************************************************************
函数：static void wmPrepHashedPatternGroups(WM_STRUCT *)
目的：计算共有多少个不同的哈希值，且从小到大
参数：
ps => 模式串集
返回：
****************************************************************/
static void wmPrepHashedPatternGroups(WM_STRUCT *ps)
{
	unsigned sindex,hindex,ningroup;
	int i;
	int m=ps->msSmallest;
	ps->msNumHashEntries=HASHTABLESIZE;	//HASH表的大小
	ps->msHash=(HASH_TYPE*)malloc(sizeof(HASH_TYPE)* ps->msNumHashEntries);	//HASH表
	if(!ps->msHash)
	{
		printf("No memory in wmPrepHashedPatternGroups()\n");
		return;
	}

	for(i=0;i<(int)ps->msNumHashEntries;i++)	//HASH表预处理初始化，全部初始化为(HASH_TYPE)-1
	{
		ps->msHash[i]=(HASH_TYPE)-1;
	}

	for(i=0;i<ps->msNumPatterns;i++)	//针对所有子串进行HASH预处理
	{
		hindex=HASH16(&ps->msPatArray[i].psPat[m-2]);	//对模式子串的最后两个字符计算哈希值（匹配）
		sindex=ps->msHash[hindex]=i;
		ningroup=1;
		//此时哈希表已经有序了
		while((++i<ps->msNumPatterns) && (hindex==HASH16(&ps->msPatArray[i].psPat[m-2])))	//找后缀相同的子串数
			ningroup++;
		ps->msNumArray[sindex]=ningroup;	//第i个子串，其后的子模式串与其后缀2字符相同子串的个数
		i--;
	}
}

/* ****************************************************************
    函数：static void wmPrepShiftTable(WM_STRUCT *)
	目的：建立shift表，算出每个字符块要移动的距离
	参数：
		ps => 模式串集
    返回：
	    
 ****************************************************************/
static void wmPrepShiftTable(WM_STRUCT *ps)
{
	int i;
	unsigned short m,k,cindex;
	unsigned shift;
	m=(unsigned short)ps->msSmallest;
	ps->msShift=(unsigned char*)malloc(SHIFTTABLESIZE*sizeof(char));
	if(!ps->msShift)
		return;

	for(i=0;i<SHIFTTABLESIZE;i++)	//初始化Shift表，初始值为最短字符串的长度
	{
		ps->msShift[i]=(unsigned)(m-2+1);
	}

	for(i=0;i<ps->msNumPatterns;i++)	//针对每个子串预处理
	{
		for(k=0;k<m-1;k++)
		{
			shift=(unsigned short)(m-2-k);
			cindex=((ps->msPatArray[i].psPat[k]<<8) | (ps->msPatArray[i].psPat[k+1]));//B为2
			if(shift < ps->msShift[cindex])
				ps->msShift[cindex] = shift;//k=m-2时，shift=0，
		}
	}
}

/* ****************************************************************
    函数：static void wmPrepPrefixTable(WM_STRUCT *)
	目的：建立Prefix表
	参数：
		ps => 模式串集
    返回：
	    无
 ****************************************************************/
static void wmPrepPrefixTable(WM_STRUCT *ps)//建立Prefix表
{
	int i;
	ps->msPrefix=(HASH_TYPE*)malloc(sizeof(HASH_TYPE)* ps->msNumPatterns);	//分配空间长度为所有子串的个数*
	if(!ps->msPrefix)
	{
		printf("No memory in wmPrepPrefixTable()\n");
		return;
	}

	for(i=0;i<ps->msNumPatterns;i++)	//哈希建立Prefix表
	{
		ps->msPrefix[i]=HASH16(ps->msPatArray[i].psPat);//对每个模式串的前缀进行哈希
	}
}

/* ****************************************************************
函数：void wmGroupMatch(WM_STRUCT *,int ,unsigned char *,unsigned char *)
	目的：后缀哈希值相同，比较前缀以及整个字符串匹配
	参数：
		ps => 模式串集
		lindex => 
		Tx => 要进行匹配的字符串序列
		T => 模式子串
    返回：
	    无
 ****************************************************************/
void wmGroupMatch(WM_STRUCT *ps,
	int lindex,//lindex为后缀哈希值相同的那些模式子串中的一个模式子串的index
	unsigned char *Tx,
	unsigned char *T)
{
	WM_PATTERN_STRUCT *patrn;
	WM_PATTERN_STRUCT *patrnEnd;
	int text_prefix;
	unsigned char *px,*qx;

	patrn=&ps->msPatArray[lindex];
	patrnEnd=patrn+ps->msNumArray[lindex];

	text_prefix=HASH16(T);


	for(;patrn<patrnEnd;patrn++)
	{
		if(ps->msPrefix[lindex++]!=text_prefix)
			continue;
		else	//如果后缀哈希值相同，则
		{
			px=patrn->psPat;	//取patrn的字串
			qx=T;
			while(*(px++)==*(qx++) && *(qx-1)!='\0');	//整个模式串进行比较
			if(*(px-1)=='\0')	//匹配到了结束位置，说明匹配成功
			{
				// printf("Match pattern \"%s\" at line %d column %d\n",patrn->psPat,nline,T-Tx+1);
				nfound++;
			}
		}
	}
}

/* ****************************************************************
    函数：int wmPrepPatterns(WM_STRUCT *ps)
	目的：对模式串集预处理，由plist得到msPatArray
	参数：
		ps => 模式串集
    返回：
	    int - 预处理成功0，失败-1
 ****************************************************************/
int wmPrepPatterns(WM_STRUCT *ps)
{
	int kk;
	WM_PATTERN_STRUCT *plist;

	ps->msPatArray=(WM_PATTERN_STRUCT*)malloc(sizeof(WM_PATTERN_STRUCT)*ps->msNumPatterns);
	if(!ps->msPatArray)
		return -1;

	ps->msNumArray=(unsigned short*)malloc(sizeof(short)*ps->msNumPatterns);
	if(!ps->msNumArray)
		return -1;

	for(kk=0,plist=ps->plist;plist!=NULL && kk<ps->msNumPatterns;plist=plist->next)
	{
		memcpy(&ps->msPatArray[kk++],plist,sizeof(WM_PATTERN_STRUCT));
	}
	sort(ps);	//哈希排序
	wmPrepHashedPatternGroups(ps);	//哈希表
	wmPrepShiftTable(ps);	//shift表
	wmPrepPrefixTable(ps);	//Prefix表
	return 0;
}

/* ****************************************************************
    函数：void wmSearch(WM_STRUCT *ps,unsigned char *Tx,int n)
	目的：字符串匹配查找
	参数：
		ps => 模式串集
		Tx => 被查找的字符串序列
		n => 被查找的字符串长度
    返回：
	    无
 ****************************************************************/
void wmSearch(WM_STRUCT *ps,unsigned char *Tx,int n)
{
	int Tleft,lindex,tshift;
	unsigned char *T,*Tend,*window;
	Tleft=n;
	Tend=Tx+n;
	if(n < ps->msSmallest)	/*被查找的字符串序列比最小模式子串还短，
							显然是不可能查找到的，直接退出*/
		return;

	for(T=Tx,window=Tx+ps->msSmallest-1;window<Tend;T++,window++,Tleft--)
	{
		tshift = ps->msShift[(*(window-1)<<8) | *window];
		while(tshift)//当tshift!=0,无匹配
		{
			window+=tshift;
			T+=tshift;
			Tleft-=tshift;
			if(window>Tend) return;
			tshift=ps->msShift[(*(window-1)<<8) | *window];
		}
		//tshift=0，表明后缀哈希值已经相同
		if((lindex=ps->msHash[(*(window-1)<<8) | *window])==(HASH_TYPE)-1) continue;
		lindex=ps->msHash[(*(window-1)<<8) | *window];
		wmGroupMatch(ps,lindex,Tx,T);//后缀哈希值相同，比较前缀及整个模式串
	}
}
/*
int main()
{
	int length,n;
	WM_STRUCT *p;  
	char keyword[MAXM]; //单词
	char str[MAXN]; //模式串
	p=wmNew();	//创建模式串集
	printf("Scanf the number of pattern words ->\n");	//输入模式串集中模式子串的个数,n
	scanf("%d", &n);
	printf("Scanf the pattern words ->\n");
	while(n --)
	{
		scanf("%s", keyword);	//输入每个模式子串
		length=strlen(keyword);
		wmAddPattern(p,keyword,length);	//新增模式子串
	}
	wmPrepPatterns(p);	//对模式串集预处理
	printf("Scanf the text string ->\n");
	scanf("%s", str);	//输入要被匹配的字符串序列
	length=strlen(str);
	wmSearch(p,str,length);
	wmFree(p);
	getchar();
	return(0);
}
*/
